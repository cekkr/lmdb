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
- Run a ≤30 minute (but with at least a full training batch completed) cheetah-only smoke ingest (`DBSLM_BACKEND=cheetah-db python3.14 src/train.py datasets/emotion_data.json --ngram-order 3 --eval-interval 2000 --json-chunk-size 250 --max-json-lines 1000`) and record decoder latency, Top-K hit rates, and command transcripts in `cheetah-db/README.md` + `studies/BENCHMARKS.md` now that the Go-side blockers are solved.
- Investigate the cheetah smoke helper hang: `scripts/start_cheetah_smoke_session.sh` + `scripts/run_cheetah_smoke.sh` currently stall on `datasets/emotion_data.json#chunk1` (see `var/eval_logs/cheetah_smoke_train_20251112-190626.log`) even though the Go server stays healthy. Capture stack traces and cheetah telemetry when it wedges so latency/top-K stats can finally land in the docs.
