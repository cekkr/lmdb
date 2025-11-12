## Completed
- Hardened the long-running ingest/smoke logging pipeline with timestamped trainer/decoder logs, cheetah telemetry mirroring, and smoke-train process tracking/timeouts.
- Unblocked the cheetah pair-trie and reducer stack: prefix-sharing keys now coexist, reducers stream payloads in chunks, and `PAIR_SCAN`/`PAIR_REDUCE` expose `next_cursor` so the Python adapter can page automatically.
- Extended the `CHEETAHDB_BENCH=1 go test -run TestCheetahDBBenchmark` harness to seed mock pair data, log pair-scan throughput, and persist snapshots under `var/eval_logs/cheetah_db_benchmark_*.log`.
- `scripts/drain_queue.py` now enforces `--max-json-lines 500`, trims queues back to `--queue-cap` (default 200), and records each run so the metrics can be mirrored into `studies/BENCHMARKS.md`.

## Active tasks
- Mirror the remaining Level 2/3 metadata (conversation stats, correction digests, bias presets) into cheetah namespaces so new processes cold-start without extra SQLite reads.
- Flesh out `cheetah-db/CONCEPTS.md` with the reducer + context-relativism contracts and add regression/test plans that cover Absolute Vector Order payloads plus the new pagination keywords.
- Wire queue-drain metrics into the CI/smoke harness so overflow alerts automatically trigger `scripts/drain_queue.py` and append the resulting metrics blob to `studies/BENCHMARKS.md` without manual intervention.
- Add a simple start/stop helper (or codify the documented `screen` snippet) around the `CHEETAH_HEADLESS=1` server launch so agents can reliably spawn/kill the Go service before rebuilding binaries.
- Run a ≤30 minute (but with at least a full training batch completed) cheetah-only smoke ingest (`DBSLM_BACKEND=cheetah-db python3.14 src/train.py datasets/emotion_data.json --ngram-order 3 --eval-interval 2000 --json-chunk-size 250 --max-json-lines 1000`) and record decoder latency, Top-K hit rates, and command transcripts in `cheetah-db/README.md` + `studies/BENCHMARKS.md` now that the Go-side blockers are solved.
