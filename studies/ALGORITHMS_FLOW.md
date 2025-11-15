# Algorithm Flow Notes (rough map for future perf/parallel work)

## Training (`src/train.py`)
- CLI inputs (selected): `inputs...`, `--db`, `--ngram-order`, `--context-dimensions`, `--recursive`, `--reset` (wipes cheetah caches), `--eval-interval/--eval-samples/--eval-dataset/--eval-pool-size`, `--json-chunk-size/--max-json-lines/--chunk-eval-percent`, `--profile-ingest`, `--decoder-* penalties`, `--backonsqlite`.
- Bootstrap: load settings → parse/validate args and context dimensions → optional metrics writer → init `DBSLMEngine` (cheetah hot-path if reachable, SQLite otherwise) → log RNG seeds.
- Input staging: `collect_files` + `iter_corpora` expand files (and JSON/NDJSON chunking) with per-chunk holdouts via `_sample_holdouts`; `IngestProfiler` can wrap ingest calls with resource snapshots.
- Ingest loop: for each `CorpusChunk`, `engine.train_from_text` runs on the main thread with progress throttling via `TrainingProgressPrinter`; totals/windows tracked.
- Evaluation: `InferenceMonitor` fires `run_inference_records` synchronously at token thresholds and for chunk hold-outs—ingest blocks while probes run. Dataset is rotated/refreshed to keep probe variety.
- Completion: report totals/hit ratios, finalize metrics, close DB.
- Future concurrency hooks: split ingest vs eval into separate processes/threads (shared DB/hot-path), batch training windows for worker pools, or overlap file parsing with DB writes; explore async metrics/log flushing to keep the ingest loop tight.

## Inference (`src/run.py`)
- CLI inputs: `--db`, `--ngram-order`, `--context-dimensions`, `--user`, `--agent`, `--conversation`, `--prompt`, `--max-turns`.
- Flow: load settings → parse dims → create `DBSLMEngine` → `ensure_conversation` (resume or start) → log context dims/conversation.
- Modes: single-prompt path (`issue_prompt`) vs interactive REPL (with `:history`, exit shortcuts). All work is synchronous on the main thread; DB is closed on exit.
- Future concurrency hooks: prefetch next response while user types (thread) or pool decoder calls if conversation fan-out emerges; consider streaming/logging callbacks to avoid blocking UI.

## Cheetah DB server (Go path)
- Startup (`main.go`): init `ResourceMonitor`, `Engine`, TCP server, optional headless mode; CLI loop otherwise.
- Command path: TCP/CLI → `Database.ExecuteCommand` parses verbs (`INSERT/READ/EDIT/DELETE`, `PAIR_*`, `SYSTEM_STATS`, `LOG_FLUSH`) → routes to table ops in `commands.go`/`database.go`.
- Concurrency: per-entry locks in CRUD, RW locks in trie ops, `PairReduce` already parallelizes payload reads with worker goroutines sized by `ResourceMonitor.RecommendedWorkers`; TCP connections handled per-goroutine.
- Future concurrency hooks: enlarge/auto-tune worker counts, parallelize `PAIR_SCAN` traversal, batch writes into worker queues, or shard databases within `Engine` for multi-core ingest; expose async command queue for remote writers.

## Tracing knobs (for correlating work)
- Python: set `LMDB_LOG_LEVEL=3` to emit v3 traces for argument parsing, cheetah resets, input discovery, evaluation thresholds, and holdout runs.
- Go: set `CHEETAH_LOG_LEVEL=3` for v3 command ingress/egress logs; use `LOG_FLUSH [limit]` to dump and clear the in-memory ring buffer of recent server logs.
