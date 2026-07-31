## Completed
- Hardened the long-running ingest/smoke logging pipeline with timestamped trainer/decoder logs, cheetah telemetry mirroring, and smoke-train process tracking/timeouts.
- Replaced the vendored `cheetah-db/` source tree with the `cekkr/cheetah` Git submodule. The launcher now uses the upstream `cheetah-server` output; generic Cheetah fixes must be committed in that repository before this project advances its submodule gitlink.
- Unblocked the cheetah pair-trie and reducer stack: prefix-sharing keys now coexist, reducers stream payloads in chunks, and `PAIR_SCAN`/`PAIR_REDUCE` expose `next_cursor` so the Python adapter can page automatically.
- Extended the `CHEETAHDB_BENCH=1 go test -run TestCheetahDBBenchmark` harness to seed mock pair data, log pair-scan throughput, and persist snapshots under `var/eval_logs/cheetah_db_benchmark_*.log`.
- `scripts/drain_queue.py` now enforces `--max-json-lines 500`, trims queues back to `--queue-cap` (default 200), and records each run so the metrics can be mirrored into `studies/BENCHMARKS.md`.
- Added screen-first helpers (`scripts/start_cheetah_server.sh`, `scripts/stop_cheetah_server.sh`, `scripts/run_cheetah_smoke.sh`, `scripts/start_cheetah_smoke_session.sh`) with a disciplined `tmux` fallback so Cheetah services and smoke runs launch with consistent timeouts/log paths across hosts.
- `ConversationMemory` + `BiasEngine` mirror Level 2 metadata (conversation stats, correction digests, bias presets) into `meta:l2:*` namespaces so cheetah can cold-start the higher-level caches without SQLite hits.
- `cheetah-db/CONCEPTS.md` now spells out the reducer/context-relativism contracts and the regression plan covering Absolute Vector Order payloads together with pagination cursors.
- `scripts/smoke_train.py`'s telemetry thread now auto-runs `scripts/drain_queue.py` when queue depth exceeds `--queue-drain-threshold` and appends a “Queue Drain (auto smoke harness)” entry (with metrics) to `studies/BENCHMARKS.md`.

- `src/train.py` now prints the cheetah hot-path endpoint plus the final Top-K hit ratio, and evaluation uses `helpers/char_tree_similarity.py` to surface `char_repeat_*` metrics, feed them into quality gating, and re-run repeats with stronger penalties.
- cheetah prediction tables now deepen context matrices with derived mean/variance/contrast/interaction layers (toggle via `CHEETAH_PREDICT_DEEPEN`) so prediction weights can react to richer context signals.
- cheetah-db pair entries now support hidden terminals with `PAIR_SET_HIDDEN` and `include_hidden=1` scan/reduce/summary filters.
- `PREDICT_INHERIT_BATCH`/`PREDICT_INHERIT_ASYNC` jobs are live and the trainer queues merged-token inheritance through them.
- Sentence punctuation splitting is now disabled by default during training, with opt-in flags/env to enable the legacy pass.
- Repaired the external Cheetah submodule at upstream commit `6866be9`: value inserts now reserve
  distinct slots atomically, pair mutations are serialized, and concurrent shared-prefix
  `PAIR_SET` traffic no longer loses keys.
- Updated the Python hot-path adapter for the current Cheetah job protocol (`JOB submit`,
  `JOB status`, `JOB fetch`) while retaining legacy reducer aliases for older servers. Reducer
  payloads now remove Cheetah's storage-transport base64 layer before DB-SLM deserialization.
- Reworked the Cheetah service/smoke helpers around bounded `screen` sessions (with disciplined
  `tmux` fallback), exact PID files, and explicit timeouts so stop operations target only the
  launched server.
- Completed a Cheetah-only `datasets/emotion_data.json` training and inference validation on
  2026-07-23: 20 records produced 380,200 tokens and 380,198 n-grams in 206.82 seconds, kept the
  adapter active, and generated a correctly framed response from the persisted Cheetah database.
  Full commands and storage/benchmark details are recorded in `studies/BENCHMARKS.md`.
- Repaired empty/misleading training evaluations: emotion context and canonical `|CTX|` lines now
  use one training/evaluation prompt composer, prompt-tag bans survive Level 2 cache and prediction
  blending, empty and `|END|`-only decodes (including spaced/case legacy variants) retry before a
  scaffold-free fallback, verbose staging
  logs distinguish decoder input from the held-out reference, and periodic probes no longer replay
  every token threshold crossed inside one large chunk. Evaluation/profile events now refresh the
  metrics JSON atomically while a run is active. MKN rebuilds now select the current relational
  ingest counts, delta-mirror them into Cheetah, and flush queued writes instead of reusing the
  first populated (seed-only) count namespace. Dependency layers remain side-channel
  evaluation/prediction records rather than serialized n-gram text, and legacy dependency-field
  generations are rejected before they reach evaluation output.

- Aligned training and inference with the new Cheetah graph context memory (`GRAPH_NODE_SET`,
  `GRAPH_NODE_GET`, `GRAPH_EDGE_SET_BATCH`, `GRAPH_RECALL`, `GRAPH_SIMILAR`, `GRAPH_TERM_INDEX`).
  Training records staged prompt/response pairs as `ctx:`/`term:` nodes joined by
  `evokes`/`precedes`/`dep_<label>` edges with the complete response sentence as a bounded node
  reference; inference seeds `GRAPH_RECALL` from the turn's context values and content words, biases
  decoding toward recalled terms, and widens the internal bias/embedding context with the hydrated
  sentences without ever showing them. Opt-in via `DBSLM_GRAPH_MEMORY=1` / `--graph-memory`.

- Moved the generic half of the Cheetah client into the submodule as a Python binder
  (`cheetah-db/binders/python/`, package `cheetah_db`): protocol codec, destination resolution,
  socket client and thread-local pool, and the `kv`/`graph`/`jobs`/`predict`/`admin` call layers,
  plus key primitives, a token vocabulary and a subclassable `CheetahDatabase`. `CheetahClient` in
  `src/db_slm/adapters/cheetah.py` now extends the binder's client through
  `src/db_slm/adapters/cheetah_binder.py`, and the commands DB-SLM previously could not speak are
  available: `PAIR_PUT_BATCH`, `DEL pairs/graph`, `GRAPH_EDGE_GET`/`NEIGHBORS`/`DEGREE`/
  `NEIGHBOR_TYPES`/`QUERY`, `PREDICT_BACKEND`/`PREDICT_BENCH`, `LOG_FLUSH`, `FILE_CHECKPOINT`,
  `CLUSTER_STATUS`, `FORK_ASSIGN`. Level 1 now preserves grouped context/count/probability/Top-K/
  continuation materializations: one paged namespace scan resolves existing rows, existing payloads
  are edited in place, and fresh rows go out in `PAIR_PUT_BATCH` pages
  (`CHEETAH_PAIR_BATCH_SIZE`, default 256) with a verified per-row fallback.

## Active tasks
- Measure whether the grouped Level 1 batch writes are worth their complexity at scale: record ingest
  wall time and request counts with `CHEETAH_PAIR_BATCH_SIZE=1` (per-row) against the default on the
  same corpus, in `studies/BENCHMARKS.md`. Upstream measured 0.94× on 2,000 rows against 256-way
  concurrent single writes, so the win here is expected to come from request/parse overhead rather
  than server throughput — and it may be nil.
- Measure whether graph context memory actually helps: run the emotion corpus with `--graph-memory`
  on and off at the same scale and record quality metrics, decoder latency, and per-chunk graph
  ingest cost in `studies/BENCHMARKS.md`. Tune `DBSLM_GRAPH_BIAS_WEIGHT`,
  `DBSLM_GRAPH_RECALL_PRECISION`, and `--graph-memory-terms` from that evidence before considering a
  non-zero default.
- Decide whether recall should consolidate. Cheetah's `GRAPH_RECALL` never writes back what it
  found, so a recurring association is re-derived every turn at the same cost. Reinforcing a
  frequently recalled edge's weight from the read path is the obvious follow-up, and is also an open
  item upstream in `cheetah-db/NEXT_STEPS.md`.
- `scripts/start_cheetah_server.sh` cannot enforce its timeout on hosts without `timeout`/`gtimeout`
  (macOS default): the `perl -e 'alarm ...'` fallback is built as a string, so the quotes are not
  re-parsed after expansion and the server dies immediately. Build the fallback as an array or
  invoke `perl` directly.
- Repair the `--backonsqlite` startup contract: `build_cheetah_adapter()` currently raises
  `SystemExit` before `train.py` can honor the flag when a configured Cheetah service is
  unreachable. Add a focused startup regression, then either make the explicit fallback work or
  remove the flag and stale compatibility branch.
- Scale the repaired Cheetah-only path from the bounded 20-record validation to an
  evaluation-enabled 250/1,000-record emotion run; record decoder latency, Top-K hit ratio, and
  quality metrics without enabling SQLite fallback.
- Validate the new deepened prediction layers against GPTeacher eval probes and log whether punctuation repetition drops; adjust `CHEETAH_PREDICT_DEEPEN` or `--cheetah-token-weight` based on the quality metrics.
- Use the new decoder scoring pipeline traces to isolate punctuation collapse (base vs cache vs prediction) and decide whether to add a dedicated punctuation-penalty stage or a run.py trace flag.
## Remember
- The development works also on Cheetah: don't fallback on sqlite if it doesn't works, but fix the issue.
