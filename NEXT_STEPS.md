1. Harden the long-running ingest/smoke logging pipeline so field troubleshooting requires zero manual scrapes (var/cheetah_smoke_ingest.log; operator logbook 2025-11-11).
   - Deliverables: every trainer/decoder log line prefixes a `+[seconds_since_start]` timestamp so throughput stalls are immediately visible.
   - Mirror cheetah-db server telemetry (latency, queue depth, process stats) into the same log bundle instead of relying on ad-hoc screen sessions.
   - Update the smoke-train harness to track/kill lingering parallel runs and enforce a wall-clock budget so aborted prompts do not leave extra processes skewing metrics.
   - Keep attention to test and smoke-train to infinity loops: they've to deliver a result after a maximum of 30 minutes


- Run a ≤30 minute (but with at least a training batch completed) cheetah-only smoke ingest (`DBSLM_BACKEND=cheetah-db python3.14 src/train.py datasets/emotion_data.json --ngram-order 3 --eval-interval 2000 --json-chunk-size 250 --max-json-lines 1000`) and record decoder latency, Top-K hit rates, and command transcripts in `cheetah-db/README.md` + `studies/BENCHMARKS.md`.
- Harden the new `PAIR_REDUCE probabilities/continuations` feeds: add pagination for >4K slices, stress test the base64 payload path, and snapshot reducer throughput vs. the old SQLite readers.
- Mirror the remaining Level 2/3 metadata (conversation stats, correction digests, bias presets) into cheetah namespaces so new processes cold-start without extra SQLite reads.
- Integrate `scripts/drain_queue.py` into the retrain workflow: wire it to the CI/smoke harness, cap the queue at 200 entries, and append throughput snapshots to `studies/BENCHMARKS.md`.
- Flesh out `cheetah-db/CONCEPTS.md` with the reducer + context-relativism contracts and add regression plans/tests covering Absolute Vector Order payloads.
