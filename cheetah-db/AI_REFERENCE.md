# Cheetah Reference

Cheetah-specific directives and operational notes live here. Refer back to this file whenever you work with `cheetah-db`.

- **Cheetah-db is now the authoritative database.** Every ingest run, decoder lookup, and evaluation
  must assume cheetah is the primary store for counts/probabilities/context metadata. SQLite survives
  only as a scratch/cache/export format (e.g., `--db var/tmp.sqlite3` for quick analysis or when
  emitting `.sqlite3` artifacts) and should never be treated as the long-term source of truth again.
  When in doubt: start/attach to the cheetah server first, keep `DBSLM_BACKEND=cheetah-db`, and only
  lean on SQLite when a workflow explicitly requires a transient file. The trainer now strictly exits
  if the cheetah TCP endpoint cannot be reached—there is no SQLite fallback path anymore.
- `README.md` now reinforces this requirement: it includes a concrete `.env` block (host/port/database
  knobs), documents the tmux helpers (`scripts/start_cheetah_server.sh`/`stop_cheetah_server.sh`),
  and explains when `python src/train.py ... --reset` or the emergency `--backonsqlite` flag should
  be used. Treat SQLite as a small/fast scratchpad only; all long-lived runs must keep
  `DBSLM_BACKEND=cheetah-db`.
- When training/decoding from inside WSL but pointing at a cheetah server running on Windows, the
  hot-path adapter auto-retries the Windows bridge IP discovered via `/etc/resolv.conf` whenever the
  configured `DBSLM_CHEETAH_HOST` resolves to loopback. Override `DBSLM_CHEETAH_HOST` with the exact
  Windows/LAN address when cheetah lives elsewhere (container, remote host, etc.). If the server
  advertises `0.0.0.0:4455`, keep the client pointed at a *real* address (127.0.0.1, LAN IP, etc.);
  connecting to `0.0.0.0` is invalid, so the adapter now rewrites that case to loopback.
- `cheetah-db` now keeps a bounded payload cache inside `database.go`, keyed by
  `<value_size, table_id, entry_id>` so hot `READ`/`PAIR_REDUCE` loops remain in RAM instead of
  pounding the same `values_<size>_<tableID>.table` sectors. It defaults to 16k entries (~64 MB) and
  is tunable via `CHEETAH_PAYLOAD_CACHE_ENTRIES`, `CHEETAH_PAYLOAD_CACHE_MB`, or
  `CHEETAH_PAYLOAD_CACHE_BYTES` (set either to `0` to disable the cache when profiling raw disk I/O).
- Context relativism is now first-class: `AbsoluteVectorOrder` deterministically sorts nested token
  structures and mirrors them into the `ctxv:` namespace, `DBSLMEngine.context_relativism()` streams
  probabilistic projections directly from cheetah, and `Decoder` falls back to those slices whenever
  a Top-K entry is missing.
- `CheetahHotPathAdapter` mirrors raw follower counts (`PAIR_REDUCE counts`) and decoder metadata so
  MKNS rebuilds and session-cache profiles can run entirely over TCP. `NGramStore.topk_hit_ratio()`
  exposes coverage so you can watch cheetah eventually serve ≥90% of decoder requests.
- Probability/backoff slices (`prob:<order>`) and continuation metadata (`cont:`) are mirrored into
  cheetah alongside counts, and the Go reducers now return inline payloads for `counts`,
  `probabilities`, and `continuations`, eliminating the extra `READ` hop per entry.
- `src/train.py` now logs the active cheetah hot-path endpoint (host, port, namespace) at startup and
  reports the observed Top-K hit ratio after ingest completes so every run leaves an explicit trace
  that cheetah-db handled the work (runs still abort unless `--backonsqlite` is provided when the
  server is unreachable).
- cheetah-db now mirrors context metadata + Top-K slices directly during ingest. `DBSLM_BACKEND`
  defaults to `cheetah-db`, the decoder reports its hit ratio via
  `DBSLMEngine.cheetah_topk_ratio()`, and Level 1 lookups can iterate namespaces with
  `NGramStore.iter_hot_context_hashes()` or trigger probabilistic tree queries via
  `engine.context_relativism(...)`. The old `ColdStorageFlusher`/MariaDB path has been removed.
- cheetah-db now keeps persistent file handles per pair-trie node (RW locked), parallelizes reducer
  payload hydration with a bounded worker pool, and treats child pointers + terminal keys as
  independent flags so prefix-sharing namespaces (`ctx:*`, `ctxv:*`, `topk:*`, etc.) finally
  coexist. `PAIR_SCAN`/`PAIR_REDUCE` accept optional cursors and emit `next_cursor=x...` when a page
  hits the configured limit, allowing clients to stream arbitrarily large namespaces without
  reopening readers. Run `CHEETAHDB_BENCH=1 go test -run TestCheetahDBBenchmark -count=1 -v` from
  `cheetah-db/` to reproduce the latest snapshots:
  - `var/eval_logs/cheetah_db_benchmark_20251112-130623.log` — 24 workers / 30 s (~64 ops/s aggregate).
  - `var/eval_logs/cheetah_db_benchmark_20251112-164324.log` — 32 workers / 45 s (90→56 ops/s before the graceful drain, 1002 inserts, errors=0).
  - `var/eval_logs/cheetah_db_benchmark_20251112-164803.log` — 24 workers / 30 s rerun (96→67 ops/s, pair scans present in every bucket).
- cheetah-server now boots with a resource monitor: it detects logical cores, samples process vs
  system CPU percentages, and polls `/proc/self/io` for disk churn. Reducer worker pools call
  `RecommendedWorkers()` so hot `PAIR_REDUCE` bursts automatically back off when CPU or I/O pressure
  spikes, keeping multi-connection workloads responsive. Issue `SYSTEM_STATS` via CLI/TCP to read
  the latest snapshot (`logical_cores`, goroutines, CPU%, bytes/sec), and feed those numbers back
  into ingest scripts when you need adaptive fan-out across multiple cheetah nodes.
- `src/train.py` and `run.py` expose `--context-dimensions`, a comma-separated list of span ranges
  (e.g., `1-2,3-5`) or progressive lengths (e.g., `4,8,4`). Length specs auto-expand to contiguous
  spans starting at 1, and logs now append `(len=...)` so you can see the effective window widths.
  Selections live in `tbl_metadata` (and the cheetah metadata mirror) so repeat penalty tracking
  persists between runs and `Decoder` can down-weight word/sentence-length sequences that still leak
  through.
- MariaDB migrations are gone. Reset SQLite tables in place (or swap DB paths) and let cheetah's
  namespaces carry the hot/archive copies—no second store to reconcile or SQL bundle to ship.
- `--reset` only unlinks the SQLite file resolved via `--db`/`DBSLM_SQLITE_PATH` (defaults to
  `var/db_slm.sqlite3`). It never renames or truncates the cheetah namespace. Pick a distinct
  `DBSLM_CHEETAH_DATABASE` per run (or run `cheetah-db` cleanup commands) when you need isolated
  hot-path data instead of relying on `--reset`.
- `Makefile` now shells into `scripts/smoke_train.py`, which can iterate arbitrary scenario matrices,
  stream live metrics, and hand each scenario a dedicated SQLite + `DBSLM_CHEETAH_DATABASE`
  namespace so cheetah sessions can be paused and restarted independently.

### cheetah-db caching & SSD-wear guideline

- Launch `cheetah-server` with an explicit payload-cache budget whenever you expect repeated
  namespace hits (ingest, MKNS rebuilds, held-out decoder runs). The defaults
  (`CHEETAH_PAYLOAD_CACHE_ENTRIES=16384`, `CHEETAH_PAYLOAD_CACHE_MB=64`) cover average corpora, but
  bump the byte budget to 128–256 MB on larger hosts to eliminate the last SSD reads against the
  `values_*` tables.
- When profiling raw disk I/O or working on RAM-starved systems, disable the cache by exporting
  `CHEETAH_PAYLOAD_CACHE_ENTRIES=0` (or the MB/bytes variants). Re-enable it immediately afterward so
  normal workloads keep leveraging RAM instead of falling back to repeated SSD seeks.

- Per-scenario environment overrides (`DBSLM_SQLITE_PATH`, `DBSLM_CHEETAH_DATABASE`) keep SQLite and
  cheetah namespaces isolated, allowing the smoke train to stop one DB session and spin up another
  without restarting the Go service. Override the cheetah namespace manually by setting
  `DBSLM_CHEETAH_DATABASE` before launching any CLI if you need ad-hoc names outside the matrix.
- New helpers:
  - `scripts/start_cheetah_server.sh` / `scripts/stop_cheetah_server.sh` wrap the tmux fallback for
    launching/stopping the Go server when `screen` cannot stay attached inside WSL.
  - `scripts/run_cheetah_smoke.sh` enforces the cheetah-only smoke flags (tmp SQLite DB, metrics
    export, timeout), and `scripts/start_cheetah_smoke_session.sh` runs it inside a tmux session so
    progress can be tailed independently of PowerShell.
  - Always monitor the emitted log; kill the `cheetah_smoke` session if the tail stops advancing.
    Current failure to triage: `var/eval_logs/cheetah_smoke_train_20251112-190626.log` sticks on
    `datasets/emotion_data.json#chunk1` even though the server shows no errors.
### cheetah-only Archive

- Start `cheetah-db/cheetah-server` before running the Python CLI. `DBSLM_BACKEND` defaults to
  `cheetah-db`, so Level 1 lookups hit the Go service automatically.
- The Python bridge keeps reducer sockets alive across long-running queries: `CheetahClient` now
  tolerates up to ~30 seconds of inactivity while waiting for `PAIR_REDUCE` responses, so heavy
  count/probability pages no longer trip the 1-second TCP timeout. Increase
  `DBSLM_CHEETAH_TIMEOUT_SECONDS` only if you truly need longer windows.
- The trainer now refuses to silently fall back to SQLite when `DBSLM_BACKEND=cheetah-db`. If the
  Go server is down or you forgot to launch `cheetah-db/cheetah-server`, `src/train.py` exits with
  an error unless you explicitly pass `--backonsqlite` (intended only for emergency smoke reruns).
  Keep the compiled server running in parallel with every ingest/smoke session to avoid wasting
  runs on the wrong backend.
- Export `CHEETAH_HEADLESS=1` when launching the server inside WSL or a Windows terminal to disable
  the interactive CLI and leave the TCP loop running in the background. Typical pattern:
  `wsl.exe -d Ubuntu-24.04 -- screen -dmS cheetahdb bash -c 'cd /mnt/c/.../cheetah-db && env CHEETAH_HEADLESS=1 ./cheetah-server-linux'`.
  Always `screen -ls`/`screen -wipe` (or `pkill -f cheetah-server`) before rebuilding so the binary
  can be replaced cleanly.
- Pair trie inserts now allow prefix-sharing keys and chunked reducers/paginators are live, so the
  cheetah-only smoke ingest backlog is unblocked. Every `PAIR_SCAN`/`PAIR_REDUCE` response carries
  `next_cursor=x...` when additional pages exist, and the Python adapter follows those cursors
  automatically (`scan_namespace`, `iter_counts`, `iter_probabilities`, `iter_continuations`), so
  you can iterate huge namespaces without custom pagination loops.
- There is no SQL migration step or MariaDB destination anymore. Reset SQLite in place when you
  need a clean rebuild; cheetah already mirrors every context/top-K slice as part of the ingest loop.
- Watch `DBSLMEngine.cheetah_topk_ratio()` (or the training log line) to confirm cache coverage stays
  ≥90% so Top-K reads rarely fall back to SQLite.
- `sqlite` is **strictly a convenience/export format** now. Use it for short-lived corpus slicing,
  ad-hoc diffs, or when emitting `.sqlite3` bundles for downstream tools, but do not ship features or
  workflows that depend on SQLite-specific behavior. If you need a clean state, blow away the SQLite
  file with `--reset` and/or pick a new cheetah namespace—never attempt to keep long-running state in
  SQLite.
- `cheetah-db` (see `cheetah-db/`) is the real database. Always ensure `cheetah-db/cheetah-server`
  is running, set `DBSLM_BACKEND=cheetah-db` (or at least `DBSLM_CHEETAH_MIRROR=1` during local
  smoke tests), and double-check every new command or script prints the cheetah namespace it targets.
  The default train command in this repo assumes cheetah is healthy; if cheetah is down you must
  either fix it or pass `--backonsqlite` with an explicit rationale recorded in `NEXT_STEPS.md`.
  As of this pass:
  - the trie exposes `PAIR_SCAN` plus `PAIR_REDUCE counts`, so MKNS rebuilds and cache coverage
    metrics stream directly from Go without materializing temporary tables in SQLite;
  - the absolute vector ordering codec (`ctxv:` namespace) allows byte-identical context relativism,
    enabling nested queries + decoder fallbacks via `engine.context_relativism()`; and
  - metadata (context dimensions, decode presets, etc.) now lives in cheetah namespaces so new
    processes can cold-start with zero SQLite reads beyond the base schema.
  Keep `NEXT_STEPS.md` updated with those gaps and record interoperability details in
  `cheetah-db/README.md` for future agents, since the roadmap now aims to delete the remaining
  SQLite-only code paths once the reducers land.
