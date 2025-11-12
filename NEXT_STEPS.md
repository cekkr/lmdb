## Completed
- Hardened the long-running ingest/smoke logging pipeline with timestamped trainer/decoder logs, cheetah telemetry mirroring, and smoke-train process tracking/timeouts.
- Unblocked the cheetah pair-trie and reducer stack: prefix-sharing keys now coexist, reducers stream payloads in chunks, and `PAIR_SCAN`/`PAIR_REDUCE` expose `next_cursor` so the Python adapter can page automatically.
- Extended the `CHEETAHDB_BENCH=1 go test -run TestCheetahDBBenchmark` harness to seed mock pair data, log pair-scan throughput, and persist snapshots under `var/eval_logs/cheetah_db_benchmark_*.log`.
- `scripts/drain_queue.py` now enforces `--max-json-lines 500`, trims queues back to `--queue-cap` (default 200), and records each run so the metrics can be mirrored into `studies/BENCHMARKS.md`.
- Added tmux-based helpers (`scripts/start_cheetah_server.sh`, `scripts/stop_cheetah_server.sh`, `scripts/run_cheetah_smoke.sh`, `scripts/start_cheetah_smoke_session.sh`) so cheetah services and smoke runs can launch with consistent timeouts/log paths even when WSL `screen` sessions are unavailable.
- `ConversationMemory` + `BiasEngine` mirror Level 2 metadata (conversation stats, correction digests, bias presets) into `meta:l2:*` namespaces so cheetah can cold-start the higher-level caches without SQLite hits.
- `cheetah-db/CONCEPTS.md` now spells out the reducer/context-relativism contracts and the regression plan covering Absolute Vector Order payloads together with pagination cursors.
- `scripts/smoke_train.py`'s telemetry thread now auto-runs `scripts/drain_queue.py` when queue depth exceeds `--queue-drain-threshold` and appends a “Queue Drain (auto smoke harness)” entry (with metrics) to `studies/BENCHMARKS.md`.

## Active tasks
- Cheetah-only smoke ingest follow-up:
  - 2025-11-12 run (tmux `cheetah_smoke`, `/tmp/db_slm_smoke.sqlite3`) hit the ≤30 minute budget and finished ingesting `datasets/emotion_data.json#chunk1`; see `var/cheetah_smoke_train_20251112-205914.log` for the transcript.
  - `Disabling cheetah hot-path adapter: pair_reduce counts failed` kept the decoder on SQLite, so the cheetah Top-K hit ratio stayed at 0% and no latency JSON flushed before the timeout. Next step: fix `PAIR_REDUCE counts` on the Go side + adapter, rerun the same command, and record the decoder latency / Top-K stats in `cheetah-db/README.md` + `studies/BENCHMARKS.md`.
- Investigate the cheetah smoke helper hang: `scripts/start_cheetah_smoke_session.sh` + `scripts/run_cheetah_smoke.sh` still wedge on `datasets/emotion_data.json#chunk1` (originally in `var/eval_logs/cheetah_smoke_train_20251112-190626.log`, now reproducible in `var/cheetah_smoke_train_20251112-205914.log` where the eval loop burns retries indefinitely). Capture stack traces and cheetah telemetry the moment it wedges so we can remove the repetitive “Zooming in…” scaffolds and finally unlock reliable latency/top-K recordings.
