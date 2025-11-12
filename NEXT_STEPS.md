## Completed
- Hardened the long-running ingest/smoke logging pipeline with timestamped trainer/decoder logs, cheetah telemetry mirroring, and smoke-train process tracking/timeouts.

## Active tasks
- Run a ≤30 minute (but with at least a training batch completed) cheetah-only smoke ingest (`DBSLM_BACKEND=cheetah-db python3.14 src/train.py datasets/emotion_data.json --ngram-order 3 --eval-interval 2000 --json-chunk-size 250 --max-json-lines 1000`) and record decoder latency, Top-K hit rates, and command transcripts in `cheetah-db/README.md` + `studies/BENCHMARKS.md`. **Blocked:** the Go pair-trie currently rejects some `PAIR_SET ctx:*` inserts and the `PAIR_REDUCE counts` stream dies with `internal_error:EOF` when payloads exceed ~60KB. Land the trie node-splitting + chunked reducers before rerunning.
- Harden the new `PAIR_REDUCE probabilities/continuations` feeds: add pagination for >4K slices, stress test the base64 payload path, and snapshot reducer throughput vs. the old SQLite readers.
- Mirror the remaining Level 2/3 metadata (conversation stats, correction digests, bias presets) into cheetah namespaces so new processes cold-start without extra SQLite reads.
- Integrate `scripts/drain_queue.py` into the retrain workflow: wire it to the CI/smoke harness, cap the queue at 200 entries, and append throughput snapshots to `studies/BENCHMARKS.md`.
- Flesh out `cheetah-db/CONCEPTS.md` with the reducer + context-relativism contracts and add regression plans/tests covering Absolute Vector Order payloads.
- Patch `cheetah-db/PairSet` (or normalize metadata key lengths) so prefix-sharing keys can coexist and update the reducer RPCs to stream large payloads in chunks. Document the fixes in `AI_REFERENCE.md` + `cheetah-db/README.md` and add regression coverage.
- Extend the new `CHEETAHDB_BENCH=1 go test -run TestCheetahDBBenchmark` harness: capture pair-scan throughput, push the logs into `studies/BENCHMARKS.md`, and trim the transient "no key/pair" errors by seeding more mock data before the timed window.
