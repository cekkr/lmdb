# AI Reference

Fast-access knowledge base for CUDAway. Update this alongside `README.md` whenever code or docs
change so the next agent inherits the latest context.

## Collaboration Rules

- Mirror meaningful changes into `README.md`, `AI_REFERENCE.md`, and (if a study) `studies/*`.
- Read and record research results under `studies/` and cross-link them here so we avoid re-running the same
  investigations.
- Work incrementally, document breaking changes, and run the available build/test commands before
  yielding control.
- Always launch long-running services and workloads (e.g., `cheetah-server`, smoke-train/benchmark runs, CI smoke tests) inside `screen` sessions: verify at the start of every screen invocation that no previous sessions are lingering, monitor the session output in real time, and attach explicit timeouts (≤30 minutes by default, ≤1 hour only if justified ahead of time) so stuck/error loops do not block the next agent. When the WSL image cannot keep `screen` alive (missing setuid bit), fall back to `tmux` with the exact same discipline and log the substitution in your notes/runbook.

### Next Steps

`NEXT_STEPS.md` is now the single backlog source; consult it for active tooling tasks and update that file first before mirroring high-level context here. Update it with your new objectives, remove already completed and no more important steps to remember.

## Codebase State

- `src/db_slm` now mirrors the v2 DB-SLM spec. Level 1 owns the vocabulary, regex tokenizer, hashed
  context registry, Modified Kneser–Ney smoother, quantized probability tables, and Top-K cache.
  Level 2 combines episodic logging, concept-ready correction digests, pointer-sentinel session
  caches, and token-level bias plumbing. Level 3 provides concept dictionaries, template
  verbalization, probability materialization, and conversation-scoped signals.
- `studies/DB_SLM_DATABASE_AND_ALGORITHMS.md` remains the authoritative schema/algorithm reference
  for the relational layout plus the KN materialization + decoding loops implemented in Python.
- `requirements.txt` now installs `sentence-transformers` (external embedding baseline) and
  `language-tool-python` (grammar deltas for the quality gate). Optional GPU acceleration is
  auto-detected via PyTorch when present.
- `src/train.py` streams corpora into the SQLite store, triggering KN rebuilds + Top-K refreshes per
  ingest; `src/run.py` exposes the concept-aware REPL that performs Level 3 → Level 1 decoding with
  cache/bias adjustments.
- Long ingest phases now emit stage-aware progress lines (vocab, each n-gram order, smoothing) so
  large JSON chunks no longer look frozen; the logs include approximate line counts to show where
  the trainer is spending time.
- Added `src/log_helpers.log`, wired it through `src/train.py`, `src/run.py`, and `src/db_slm` helpers, and now every trainer/decoder line (including telemetry emitted by `scripts/smoke_train.py`) is prefixed with `+[seconds_since_start]`; the smoke-train harness also mirrors cheetah latency/queue stats, tracks lingering subprocesses, and enforces 30-minute budgets before logging the same bundle to `var/cheetah_smoke_ingest.log`.
- `src/db_slm/sentence_parts.py` feeds `DBSLMEngine.train_from_text()` with punctuation-aware
  segments, embedding signatures, and emotion keyword tokens so Level 1 learns efficient splits in
  real time. Configure the embedding backbone with `DBSLM_EMBEDDER_MODEL`, or force hashed-only,
  offline guidance (no Hugging Face downloads) via `DBSLM_EMBEDDER_OFFLINE=1`.
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
- Evaluation probes now request at least 20 generated words (scaling up toward the reference length)
  via a response backstop so lexical / ROUGE / perplexity logs never drop a row due to blank or
  truncated generations.
- cheetah-db now mirrors context metadata + Top-K slices directly during ingest. `DBSLM_BACKEND`
  defaults to `cheetah-db`, the decoder reports its hit ratio via
  `DBSLMEngine.cheetah_topk_ratio()`, and Level 1 lookups can iterate namespaces with
  `NGramStore.iter_hot_context_hashes()` or trigger probabilistic tree queries via
  `engine.context_relativism(...)`. The old `ColdStorageFlusher`/MariaDB path has been removed.
- Level 2 metadata now rides the same channel: `ConversationMemory` writes stats to
  `meta:l2:stats:<conversation_id>`, correction digests to `meta:l2:corr:<conversation_id>`, and
  bias presets to `meta:l2:bias:<conversation_id|__global__>`. Decoder/cache components consult those
  JSON blobs first so a restarted trainer no longer needs warm-up SQL reads before issuing concept or
  bias-aware generations.
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
- `src/train.py` now exposes `--profile-ingest` for RSS/latency logging and prints lexical overlap,
  ROUGE-L, plus generated/reference perplexity in every evaluation probe so we can quantify gains
  during long streaming ingests.
- `src/train.py` now accepts `--decoder-presence-penalty` and `--decoder-frequency-penalty` so repeat
  penalty grids can run directly from the CLI; overrides propagate to periodic + hold-out probes and
  are recorded inside the metrics metadata for downstream comparisons.
- `src/train.py` and `run.py` expose `--context-dimensions`, a comma-separated list of token span
  ranges (default `1-2,3-5`) that extend presence/frequency penalties to grouped tokens. The selected
  spans are persisted inside `tbl_metadata`, automatically loaded by `DBSLMEngine`, and the decoder
  now down-weights candidates that would recreate overused word- or sentence-length sequences.
- `src/train.py` can reserve a slice of every JSON/NDJSON chunk for immediate evaluation via
  `--chunk-eval-percent`; those hold-out prompts/responses skip training, run through the same
  inference metrics the moment the chunk finishes ingesting, and refresh the rolling evaluation pool
  so future periodic probes always contain freshly sampled rows instead of the initial fixed set.
- Training-time probes now spin up seedless conversations so low-resource scaffolding never masks
  evaluation outputs; if you still need the canned turns (e.g., via `run.py`) the helper stays
  enabled for interactive sessions only.
- `DBSLMEngine` still seeds low-resource conversations with two caretaker turns, but the paraphraser
  now uses length-aware thresholds and explicit guard rails so multi-turn prompts or corrective
  instructions are never rewritten while we avoid verbatim echoes.
- Responses produced via `train.py` probes and `run.py` REPL now emit tagged frames
  (`|USER|`, `|RESPONSE|`, and similarly `|TAG_NAME|`) with randomized keyword-focused openers so evaluation logs and
  training data never echo prompts verbatim and clearly distinguish generated text from context.
- Evaluation summaries are written both to stdout and to structured JSON under
  `var/eval_logs/train-*.json`. Set `--metrics-export <path>` (or `-` to disable) to control the feed,
  which captures probe averages plus optional ingest profiling samples.
- Context-dimension runs automatically score every held-out prompt twice per probe, logging variants
  `#idx.1`/`#idx.2` separately so the grouped frequency penalty can adapt in real time and surface
  duplicate structures that still leak through.
- Sentence quality checks now combine LanguageTool grammar deltas, the CoLA acceptability head, and
  embedder-based semantic similarity/novelty. Metrics land next to lexical/ROUGE/perplexity in the
  eval logs, and low-scoring generations are appended to `DBSLM_QUALITY_QUEUE_PATH`
  (`var/eval_logs/quality_retrain_queue.jsonl` by default) so we can re-train against the weakest
  samples later.
- Evaluation probes now emit structural-diversity metrics (`structure_variety`, `common_token_penalty`,
  `token_group_share`, `top_token_share`, opener diversity, punctuation balance) that explicitly devalue templated
  responses. `QualityGate` ingests the same metrics so over-repeated openings or punctuation abuse
  get queued for retraining, and both periodic and chunk hold-out probes always run at least two
  samples (topped up from the rolling pool when needed). `SentenceQualityScorer` now scales
  `quality_score` down whenever `token_group_share` exceeds 0.30 so repetition spikes show up in the
  aggregate means.
- Latest smoke-train matrix (2025-11-10, python3.11) now runs
  `baseline_profiled` (400-row ingest, profiling enabled) followed by
  `penalty_sweep_holdout` (240-row ingest with chunk hold-outs + penalty overrides) via
  `scripts/smoke_train.py`. Combined they logged ~882k tokens with avg quality 0.599 and
  structure_variety 0.317 (`studies/BENCHMARKS.md`). Real-time stats stream into
  `var/smoke_train/benchmarks.json`, while full evaluation payloads land in
  `var/smoke_train/metrics/<scenario>.json`. Monitor the 64% flagged rate—pool diversity or penalty
  tuning is still needed to push it below the new 0.55 `common_token_ceiling`.
- Evaluation retries for flagged samples are now capped at two attempts per batch, with flagged rows
  re-queued into a random spot of the current probe before being eligible for up to three additional
  appearances in later random batches so probes cannot loop forever when the generator keeps
  emitting low-quality responses.
- The evaluator infers `min_response_words` from the reference length (capped at 512) so long-form
  corpora like `emotion_data.json` do not lose the substantive portion of the `|RESPONSE|` frame, and
  CPU-heavy quality scoring is gated behind the adaptive load monitor to avoid starving ingestion.
- Both `train.py` and `run.py` now rely on `db_slm.inference_shared.issue_prompt()` so scripted probes
  and the REPL reuse the same conversation bootstrapper.
- MariaDB migrations are gone. Reset SQLite tables in place (or swap DB paths) and let cheetah's
  namespaces carry the hot/archive copies—no second store to reconcile or SQL bundle to ship.
- `scripts/run_paraphraser_regression.py` consumes `studies/paraphraser_regression.jsonl` to ensure
  multi-turn corrective threads, structural tags, and ordinary prompts all trigger the expected
  paraphraser behavior.
- `Makefile` now shells into `scripts/smoke_train.py`, which can iterate arbitrary scenario matrices,
  stream live metrics, and hand each scenario a dedicated SQLite + `DBSLM_CHEETAH_DATABASE`
  namespace so cheetah sessions can be paused and restarted independently.

## Operational Notes

### Streaming Ingest / Profiling

- Enable `--profile-ingest` whenever you push `--json-chunk-size` above 500 rows. The profiler logs
  per-corpus latency plus RSS deltas so you can note the tipping point in `NEXT_STEPS.md` (current
  guidance: stay under ~2.5 GB RSS and <5 s per chunk on 16 GB laptops).
- While testing queue drains or ad-hoc corpora on WSL/Windows hosts, force `--max-json-lines 500`
  (the drain helper already does this) so every run exercises the same bounded chunk path and keeps
  memory spikes predictable.
- Let the held-out probes run with ROUGE/perplexity enabled so you can correlate throughput tweaks
  with quantitative gains instead of relying on overlap logs only.
- `datasets.md` now tracks basic stats for `emotion_data.json` (avg response 347 words, max 1,251) so
  chunk sizes, eval thresholds, and paraphraser guard rails stay grounded in the actual corpora.
- Prefer `make smoke-train` for regressions: it now iterates both baseline + penalty scenarios,
  writes live stats to `var/smoke_train/benchmarks.json`, and exposes switches (see `SMOKE_*` vars in
  the `Makefile`) so you can resume or subset the matrix without editing the script.

### Queue-Drain Runs

- `scripts/drain_queue.py` automates the `Queue-Drain Retrain` preset from `studies/best_commands.md`.
  It inspects `DBSLM_QUALITY_QUEUE_PATH`, skips execution until the line count exceeds the provided
  `--threshold` (default 150), then shells out to `python3.14 src/train.py ...` with the documented
  flags. The helper now forces `--max-json-lines 500` for every drain so we exercise the same chunk
  boundaries during testing, exports metrics to
  `var/eval_logs/train-queue-drain-*.json`, and trims the queue back to `--queue-cap` entries
  (default 200) once a run succeeds.
- The smoke-train harness watches queue depth too: passing `--queue-drain-threshold`/`--queue-drain-cooldown`
  (defaults 175 / 900s) arms an automated drain worker that launches the helper as soon as telemetry
  crosses the threshold, writes the resulting metrics file under `var/smoke_train/drains/`, and
  appends a summary section (“Queue Drain (auto smoke harness) …”) to `studies/BENCHMARKS.md`.
- `--dry-run` prints the exact command; `--queue /path/to/file` and `--python` let you point at
  alternate queues/interpreters, and `--max-json-lines`/`--queue-cap` can be overridden when
  load-testing different limits. Use the PowerShell helper pattern
  `wsl.exe -d Ubuntu-24.04 -- PYTHONPATH=src ... scripts/drain_queue.py ...` whenever you need the
  Linux toolchain but want to orchestrate runs from Windows.

### Smoke-Train Matrix

- `scripts/smoke_train.py` orchestrates sequential scenarios. Default entries (`baseline_profiled`
  and `penalty_sweep_holdout`) can be overridden via `--matrix path/to/matrix.json` where the JSON
  contains either `{"scenarios": [...]}` or a plain list.
- The orchestrator tails trainer stdout and writes progress/last-log snapshots plus the most recent
  metrics summaries into `var/smoke_train/benchmarks.json`. Agents looking for real-time signals
  should watch this file instead of parsing console output.
- Each scenario automatically exports `--metrics-export` data to
  `var/smoke_train/metrics/<scenario>.json`, so downstream analysis can pull structured summaries as
  soon as a scenario finishes. The benchmark file records those paths per scenario.
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
- The telemetry thread now exposes `--queue-drain-threshold`, `--queue-drain-cooldown`,
  `--queue-drain-{script,metrics-dir,benchmarks}`, and `--disable-auto-queue-drain`. Leaving the
  automation enabled means queue overflows trigger `scripts/drain_queue.py` automatically and the
  resulting metrics block is mirrored into `studies/BENCHMARKS.md` without manual editing.
- Use `--scenarios a,b` or `SMOKE_SCENARIOS=a,b` to run a subset, `--resume-from name` to skip ahead,
  and `--dry-run` to print the commands while still updating `benchmarks.json` for planning.

### cheetah-only Archive

- Start `cheetah-db/cheetah-server` before running the Python CLI. `DBSLM_BACKEND` defaults to
  `cheetah-db`, so Level 1 lookups hit the Go service automatically.
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
- Use `engine.iter_hot_context_hashes()` for namespace sweeps and
  `engine.context_relativism([...])` when you need probabilistic trie traversals with deterministic
  ordering. Both use `PAIR_SCAN` under the hood so traversals never touch SQL.

### DB Adapters

- `sqlite` remains the compatibility backend exposed through `DBSLMSettings.backend`, but it is now
  considered legacy. Keep it only for metadata bootstrapping while the Level 1 pipelines move over
  to cheetah; new work should avoid adding features that would be SQLite-only and instead target the
  Go engine.
- `cheetah-db` (see `cheetah-db/`) now doubles as the hot-path mirror for contexts, Top-K slices, and
  raw follower counts. Keeping `DBSLM_BACKEND=cheetah-db` (or leaving the backend as `sqlite` and
  enabling `DBSLM_CHEETAH_MIRROR=1`) makes the trainer push every newly discovered context and MKNS
  Top-K bucket into the Go service via its TCP commands; the decoder then queries cheetah first and
  falls back to SQLite only when a key is missing. As of this pass:
  - the trie exposes `PAIR_SCAN` plus `PAIR_REDUCE counts`, so MKNS rebuilds and cache coverage
    metrics stream directly from Go without materializing temporary tables in SQLite;
  - the absolute vector ordering codec (`ctxv:` namespace) allows byte-identical context relativism,
    enabling nested queries + decoder fallbacks via `engine.context_relativism()`; and
  - metadata (context dimensions, decode presets, etc.) now lives in cheetah namespaces so new
    processes can cold-start with zero SQLite reads beyond the base schema.
  Keep `NEXT_STEPS.md` updated with those gaps and record interoperability details in
  `cheetah-db/README.md` for future agents, since the roadmap now aims to delete the remaining
  SQLite-only code paths once the reducers land.
