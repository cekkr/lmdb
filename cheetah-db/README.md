# cheetah-db

Ultra-rapid key/value database engine being refit to serve as the low-latency adapter for the LMDB
Python project (`src/db_slm`). The original “cheetah” fork already implements a byte-oriented storage
engine in Go; this directory tracks the work needed to evolve it into a drop-in backend (`backend =
cheetah-db`) for the DB-SLM stack.

## Why it exists

- **Bridge for extreme-latency workloads.** SQLite remains the reference implementation, but it tops
  out when Level 1 counts, caches, and Level 2 conversation traces all compete for the same file.
  `cheetah-db` is meant to absorb those hot paths with a persistent, mmap-friendly layout and a
  purpose-built TCP/CLI front end.
- **Byte-faithful keying.** Each entry is cataloged by byte length and stored inside homogenous
  tables (`values_<size>_<tableID>.table`), keeping lookups O(1) with deterministic offsets.
- **Tree-indexed pair table.** The `pairs/` directory pre-allocates 256-entry pages that behave like
  a trie. This powers rapid pair/sequence lookups and will become the substrate for context hashes
  and vector-matrix anchors once the adapter is wired into DB-SLM.

## Current architecture snapshot

- `main.go` boots both a TCP server (`0.0.0.0:4455`) and a local CLI, multiplexing commands across
  multiple logical databases (`engine.GetDatabase(name)`).
- `engine.go` lazily instantiates `Database` handles under `cheetah_data/<dbname>` and guarantees
  graceful shutdown of every open table.
- `database.go` orchestrates CRUD operations, recycling, and the pair trie:
  - `MainKeysTable` (`main_keys.table`) stores the 6-byte entries that describe value size +
    location.
  - `ValuesTable` files store fixed-width blobs grouped by byte length and table ID; offsets are
    computed by entry index so reads never have to scan variable-length segments.
  - `RecycleTable` files keep tombstoned slots per byte length so inserts can reuse space without
    compacting.
  - `PairTable` nodes under `pairs/` encode byte-wise trees whose leaves can store absolute keys or
    child pointers, giving us a hardware-friendly trie for contextual lookups. Each node now keeps a
    persistent file descriptor guarded by RW locks so concurrent scans/reducers no longer thrash the
    OS with `OpenFile` calls. Terminal flags and child pointers are stored independently, so
    prefix-sharing keys (e.g., `ctx:`, `ctxv:`, `topk:` namespaces) coexist without conflict.
    `PAIR_SCAN`/`PAIR_REDUCE` accept optional cursors and emit `next_cursor=x...` whenever more data
    is available, allowing clients to page through arbitrarily large namespaces. Reducer payloads
    are streamed in chunks, eliminating the `internal_error:EOF` failures that previously appeared
    around ~60 KB payloads.
- `server.go` accepts newline-delimited commands over TCP (same grammar as the CLI) so we can script
  adapters before we embed the engine directly.

## Adapter goals for DB-SLM

To register `cheetah-db` as a DB adapter alongside SQLite and MariaDB (`DBSLMSettings.backend`),
the engine must grow the following behaviors:

1. **Rapid statistical computation + caching.** Level 1 relies on large fan-out aggregations (counts,
   continuation tables, quantized probabilities, Top-K ranks). `cheetah-db` must expose batched
   reducers or server-side stored procedures so the Python side can trigger recomputation without
   copying data back to SQLite first.
2. **Byte-based contextualization.** DB-SLM contexts are byte-tokenized hashes. The pair tables need
   to surface APIs that can auto-sort dynamic sets (e.g., context hashes → next token IDs) so
   probabilities can be refined by traversing deeper into the trie without retraining.
3. **Multi-file brute-force sweeps.** Level 2/3 materializations often need to scan entire matrix
   orderings. The engine must coordinate scans across parallel files, stream results, and keep hot
   references stable even when data is sharded across disks.

Document every milestone in `AI_REFERENCE.md` and flag missing capabilities in `NEXT_STEPS.md` so the
adapter status stays visible to future maintainers.

## Running the server headlessly

- Export `CHEETAH_HEADLESS=1` to disable the interactive CLI and keep only the TCP loop alive. This
  is ideal when the server needs to run alongside automated tests or CI jobs.
- On Windows + WSL, launch the Linux binary with `screen` so it stays in the background:
  ```
  wsl.exe -d Ubuntu-24.04 -- screen -dmS cheetahdb bash -c 'cd /mnt/c/Sources/GitHub/lmdb/cheetah-db && env CHEETAH_HEADLESS=1 ./cheetah-server-linux'
  ```
  `screen -ls` shows the session; `screen -wipe` (or `pkill -f cheetah-server`) shuts it down before
  you rebuild.
- When `screen` is unavailable (WSL often lacks the setuid bit), use the tmux helper scripts instead:
  `scripts/start_cheetah_server.sh` spawns the `cheetahdb` session and appends logs to
  `var/cheetah-server-linux.log`, while `scripts/stop_cheetah_server.sh` kills the tmux session and
  any stray `cheetah-server` PIDs.
- The Windows binary (`cheetah-server.exe`) honors the same `CHEETAH_HEADLESS=1` toggle if you prefer
  to run directly from PowerShell.

## Smoke ingest helper

- `scripts/run_cheetah_smoke.sh` standardizes the cheetah-backed smoke train flags. It automatically
  points SQLite at `/tmp/cheetah_smoke*.sqlite3`, emits logs/metrics under
  `var/eval_logs/cheetah_smoke_train_*.{log,json}`, and enforces a configurable timeout
  (`CHEETAH_SMOKE_TIMEOUT`, defaults to 30 minutes).
- `scripts/start_cheetah_smoke_session.sh` wraps the smoke run in a dedicated tmux session so the
  ingest continues in the background. The script kills any prior `cheetah_smoke` session, spawns the
  helper above with freshly generated paths, and prints the absolute log file location so you can
  `tail -f` from PowerShell.
- Always watch the log tail while the session runs. If the log stops advancing (e.g., the current
  run in `var/eval_logs/cheetah_smoke_train_20251112-190626.log` has been stuck on
  `datasets/emotion_data.json#chunk1` for several minutes), kill the `cheetah_smoke` tmux session,
  capture the timestamp + cheetah server log, and record the failure before retrying.

## Python bridge status

- `src/db_slm` now includes a `cheetah-db` hot-path adapter. Set `DBSLM_BACKEND=cheetah-db` (or
  `DBSLM_CHEETAH_MIRROR=1` to mirror without switching the primary backend) and start the Go server
  before running `src/train.py`/`src/run.py`.
- The trainer publishes every newly discovered context plus the Top-K quantized probabilities
  produced by the MKNS smoother through the TCP API (`INSERT`, `PAIR_SET`, etc.), so low-latency
  reads land in cheetah while SQLite remains authoritative for schema-wide updates.
- The decoder consults the cheetah mirror first when sampling candidates; if the Go service is
  offline or missing a context, it automatically falls back to SQLite. This gives us byte-faithful
  keying and immediate Top-K slices with no additional SQL load.
- Ordered trie streaming is now available via the `PAIR_SCAN` command and the
  `HotPathAdapter.scan_namespace()` helper, so DB-SLM can iterate over namespaces (contexts, cached
  Top-K buckets, etc.) without touching SQLite. The adapter now follows `next_cursor` tokens
  automatically, so reducers and namespace walks page through arbitrarily large slices without
  custom tooling.
- Absolute vector ordering is live: each context now gets a deterministic `ctxv:` alias derived from
  the nested token structure. `engine.context_relativism()` streams the corresponding contexts +
  ranked continuations directly from `PAIR_SCAN ctxv:`, and the decoder uses those slices whenever
  a Top-K entry is missing.
- `PAIR_REDUCE counts` aggregates follower counts in-place so MKNS rebuilds can pull entire context
  registries through TCP. The Python smoother mirrors those projections back into `cnt:<order>`
  namespaces after every rebuild, keeping Go + SQLite in sync without extra SQL.
- Quantized probability/backoff rows (`prob:<order>`) and continuation metadata (`cont:`) are now
  mirrored alongside counts, so reducers can fetch everything needed for MKNS without touching
  SQLite.
- Metadata (context dimensions, decode profiles, cache lambdas) is persisted under the `meta:`
  namespace so new Python processes can cold-start without re-reading SQLite tables. Level 2
  conversation stats, correction digests, and bias presets surface as `meta:l2:*` keys, and the
  Python helpers normalize duplicate `meta:` prefixes so older mirrors (`meta:meta:l2:*`) remain
  readable while new writes stay canonical.
- Upcoming roadmap items (statistical reducers, ordered trie slices) should extend the same adapter
  so DB-SLM can eventually run entirely on the Go engine once Level 2/3 tables get equivalents.

## Build & run

```bash
cd cheetah-db
go test ./...        # once tests exist
bash build.sh        # produces ./cheetah-server
./cheetah-server     # starts TCP server + interactive CLI
```

Example CLI session:

```
[cheetah_data/default]> INSERT:5 hello
SUCCESS,key=1
[cheetah_data/default]> READ 1
SUCCESS,size=5,value=hello
[cheetah_data/default]> PAIR_SCAN ctx: 5
SUCCESS,count=1,items=6374783a:4
```

Over TCP you can send the same commands (newline-terminated). Use `DATABASE <name>` to switch logical
stores in either mode.

### Streaming helpers

- `PAIR_SCAN <prefix> [limit]` returns namespace-ordered slices without SQLite. Use `*` as the prefix
  to stream the entire trie or `x<HEX>` to target binary prefixes. Responses include `items=...`
  with `hex_value:abs_key` pairs so clients can fetch the referenced payloads via `READ`.
- The Python adapter exposes this via `CheetahClient.pair_scan()` and
  `HotPathAdapter.scan_namespace()`, which automatically expand namespace strings (e.g., `ctx:`) and
  strip the prefix from the returned values.

### Reducer hooks

- `PAIR_REDUCE <mode> <prefix> [limit]` piggybacks on the pair iterator but now streams inline
  payloads (`items=<key_hex>:<abs_key>:<base64_payload>`). Reducers fan out payload reads across a
  bounded worker pool to overlap disk I/O with encoding work, so they avoid the single-goroutine
  bottleneck and still skip the follow-up `READ` hop.
- Implemented modes:
  - `PAIR_REDUCE counts cnt:<order>` → follower counts serialized via `CheetahSerializer.encode_counts`.
  - `PAIR_REDUCE probabilities prob:<order>` → quantized log-probabilities + backoff alphas.
  - `PAIR_REDUCE continuations cont:` → token-level continuation metadata (`num_contexts`).
- Future reducers (`PAIR_REDUCE queue-drain`, session-cache snapshots, etc.) should follow the same
  union-by-mode pattern. Update `cheetah-db/CONCEPTS.md` when adding new modes so the RPC contract
  remains easy to discover from the Go side.

## Benchmark harness

Use the built-in load generator to run a 30-second mock ingest/lookup cycle:

```bash
cd cheetah-db
CHEETAHDB_BENCH=1 go test -run TestCheetahDBBenchmark -count=1 -v
```

The test spins up its own scratch database under `cheetah_data/bench_perf`, spawns `runtime.NumCPU()`
workers (override via `CHEETAHDB_BENCH_WORKERS`), and records 5-second snapshots in
`var/eval_logs/cheetah_db_benchmark_<timestamp>.log`.

Latest run (2025-11-12, 30s, 24 workers, 256-byte payloads) produced:

- `inserts=616` (`~25/s`)
- `reads=509` (`~20/s`)
- `pair_set=258` (`~10/s`)
- `pair_get=158` (`~6/s`)
- `pair_scan=62` (`~2/s`) — new in this run so pagination stays exercised
- `errors=0` (seeding key/pair warmups removed the transient “not found” spikes)
- total throughput `~64 ops/s`

See `var/eval_logs/cheetah_db_benchmark_20251112-130623.log` for the full timeline (including the
massively throttled tail when the benchmark idles before shutdown). Override
`CHEETAHDB_BENCH_DURATION`/`CHEETAHDB_BENCH_WORKERS` to reproduce alternative mixes.

Additional snapshots:

- `var/eval_logs/cheetah_db_benchmark_20251112-164324.log` — 45 s run with 32 workers. Total_qps
  stayed above 56 ops/s through the 40 s mark (peaking at 90.4 ops/s @5 s) before falling to 10.9
  during the graceful shutdown; final counts 1002/663/346/265/123 (insert/read/pair ops) with 0
  errors.
- `var/eval_logs/cheetah_db_benchmark_20251112-164803.log` — 30 s rerun at 24 workers. Mirrors the
  earlier throughput curve (95.6 → 66.7 ops/s across the first 25 s) while proving the pagination
  fix (pair scans show up in each window, zero reducer EOFs).

## Directory map

- `commands.go` - user-facing operations (INSERT/READ/EDIT/DELETE/PAIRSET/PAIRGET/etc.).
- `tables.go` - on-disk table implementations (`MainKeysTable`, `ValuesTable`, `RecycleTable`,
  `PairTable`).
- `types.go` – constants and binary layouts shared across tables.
- `helpers.go` – low-level utilities (value allocations, binary encoding helpers).
- `server.go` / `main.go` – TCP server + CLI wiring.

## Next steps

- Mirror Level 2/3 metadata (conversation stats, bias presets, correction digests) into cheetah
  namespaces so cold starts avoid SQLite entirely.
- Integrate `scripts/drain_queue.py` into the smoke/CI harness and record queue throughput snapshots
  under `studies/BENCHMARKS.md`.
- Capture decoder latency + Top-K hit-rate snapshots in this README after each cheetah-only smoke
  train so we can watch coverage trend toward 100%.
- Add regression tests around absolute-vector ordering, reducer semantics, and new namespaces to
  guard the binary layouts before we start inlining cheetah inside the Python repo.
