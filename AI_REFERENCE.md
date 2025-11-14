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

- **Cheetah-db is now the authoritative database.** Every ingest run, decoder lookup, and evaluation
  must assume cheetah is the primary store for counts/probabilities/context metadata. SQLite survives
  only as a scratch/cache/export format (e.g., `--db var/tmp.sqlite3` for quick analysis or when
  emitting `.sqlite3` artifacts) and should never be treated as the long-term source of truth again.
  When in doubt: start/attach to the cheetah server first, keep `DBSLM_BACKEND=cheetah-db`, and only
  lean on SQLite when a workflow explicitly requires a transient file. The trainer now strictly exits
  if the cheetah TCP endpoint cannot be reached—there is no SQLite fallback path anymore.
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
- `src/db_slm` now mirrors the v2 DB-SLM spec. Level 1 owns the vocabulary, tokenizer (regex by
  default or Hugging Face `tokenizers` when configured), hashed
  context registry, Modified Kneser–Ney smoother, quantized probability tables, and Top-K cache.
  Level 2 combines episodic logging, concept-ready correction digests, pointer-sentinel session
  caches, and token-level bias plumbing. Level 3 provides concept dictionaries, template
  verbalization, probability materialization, and conversation-scoped signals.
- `studies/DB_SLM_DATABASE_AND_ALGORITHMS.md` remains the authoritative schema/algorithm reference
  for the relational layout plus the KN materialization + decoding loops implemented in Python.
- `requirements.txt` now installs `sentence-transformers` (external embedding baseline),
  `language-tool-python` (grammar deltas for the quality gate), and Hugging Face `tokenizers`
  (optional but enabled by default for the new tokenizer backend). Optional GPU acceleration is
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
  real time. Emotion tags are now derived solely from the segment keywords + embedder energy
  (the `_EMOTION_WORDS` allowlist is gone), so corpora dictate which affective tokens show up. Configure
  the embedding backbone with `DBSLM_EMBEDDER_MODEL`, or force hashed-only, offline guidance (no
  Hugging Face downloads) via `DBSLM_EMBEDDER_OFFLINE=1`.
- Tokenization now supports a Hugging Face-backed backend: set
  `DBSLM_TOKENIZER_BACKEND=huggingface`, point `DBSLM_TOKENIZER_JSON` at a tokenizer.json
  (usually exported from `tokenizers` or HF Hub), and optionally disable lower-casing with
  `DBSLM_TOKENIZER_LOWERCASE=0`. Missing packages or files trigger a logged warning and fall back to
  the legacy regex splitter so training/evals always proceed.
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
- `src/train.py` now logs the active cheetah hot-path endpoint (host, port, namespace) at startup and
  reports the observed Top-K hit ratio after ingest completes so every run leaves an explicit trace
  that cheetah-db handled the work (runs still abort unless `--backonsqlite` is provided when the
  server is unreachable).
- Evaluation now tracks cross-sample repetition via `helpers/char_tree_similarity.py`, exposes
  `char_repeat_max/avg` metrics, feeds them into `QualityGate`, and automatically re-queues variants
  with stronger presence/frequency penalties whenever the generated text matches earlier prompts too
  closely.
- cheetah-db now mirrors context metadata + Top-K slices directly during ingest. `DBSLM_BACKEND`
  defaults to `cheetah-db`, the decoder reports its hit ratio via
  `DBSLMEngine.cheetah_topk_ratio()`, and Level 1 lookups can iterate namespaces with
  `NGramStore.iter_hot_context_hashes()` or trigger probabilistic tree queries via
  `engine.context_relativism(...)`. The old `ColdStorageFlusher`/MariaDB path has been removed.
- Level 2 metadata now rides the same channel: `ConversationMemory` writes stats to
  `meta:l2:stats:<conversation_id>`, correction digests to `meta:l2:corr:<conversation_id>`, and
  bias presets to `meta:l2:bias:<conversation_id|__global__>`. Decoder/cache components consult those
  JSON blobs first so a restarted trainer no longer needs warm-up SQL reads before issuing concept or
  bias-aware generations. Metadata helpers now strip duplicate `meta:` prefixes and fall back to
  legacy `meta:meta:l2:*` entries so existing mirrors stay readable while new writes remain canonical.
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
  are recorded inside the metrics metadata for downstream comparisons. These knobs only influence
  evaluation decoding—training statistics still reflect the raw corpus, so use higher penalties plus
  richer context dimensions when repetition creeps into probe outputs.
- `src/train.py` and `run.py` expose `--context-dimensions`, a comma-separated list of span ranges
  (e.g., `1-2,3-5`) or progressive lengths (e.g., `4,8,4`). Length specs auto-expand to contiguous
  spans starting at 1, and logs now append `(len=...)` so you can see the effective window widths.
  Selections live in `tbl_metadata` (and the cheetah metadata mirror) so repeat penalty tracking
  persists between runs and `Decoder` can down-weight word/sentence-length sequences that still leak
  through.
- `src/train.py` can reserve a slice of every JSON/NDJSON chunk for immediate evaluation via
  `--chunk-eval-percent`; those hold-out prompts/responses skip training, run through the same
  inference metrics the moment the chunk finishes ingesting, and refresh the rolling evaluation pool
  so future periodic probes always contain freshly sampled rows instead of the initial fixed set.
- Randomness controls now live on the CLI: `--seed` pins Python's RNG for chunk sampling/hold-outs,
  `--eval-seed` sets the base for evaluation randomness (auto-generating one per run when omitted),
  and `--eval-variants` forces multiple generations per prompt even when context dimensions are off.
  `VariantSeedPlanner` derives unique sub-seeds for every prompt/variant/attempt combination so
  repeated probes explore different structures while remaining reproducible when the base seed is
  provided.
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
- `--reset` only unlinks the SQLite file resolved via `--db`/`DBSLM_SQLITE_PATH` (defaults to
  `var/db_slm.sqlite3`). It never renames or truncates the cheetah namespace. Pick a distinct
  `DBSLM_CHEETAH_DATABASE` per run (or run `cheetah-db` cleanup commands) when you need isolated
  hot-path data instead of relying on `--reset`.
- `scripts/run_paraphraser_regression.py` consumes `studies/paraphraser_regression.jsonl` to ensure
  multi-turn corrective threads, structural tags, and ordinary prompts all trigger the expected
  paraphraser behavior.
- `Makefile` now shells into `scripts/smoke_train.py`, which can iterate arbitrary scenario matrices,
  stream live metrics, and hand each scenario a dedicated SQLite + `DBSLM_CHEETAH_DATABASE`
  namespace so cheetah sessions can be paused and restarted independently.

### cheetah-db caching & SSD-wear guideline

- Launch `cheetah-server` with an explicit payload-cache budget whenever you expect repeated
  namespace hits (ingest, MKNS rebuilds, held-out decoder runs). The defaults
  (`CHEETAH_PAYLOAD_CACHE_ENTRIES=16384`, `CHEETAH_PAYLOAD_CACHE_MB=64`) cover average corpora, but
  bump the byte budget to 128–256 MB on larger hosts to eliminate the last SSD reads against the
  `values_*` tables.
- Inserts/edits now seed the cache and deletes invalidate their slots, so you can chain
  ingest → reducer → decoder without reopening the same offsets. Keep the server alive between
  pipeline stages to retain the warm cache; restarting the process will cold-start the cache and
  briefly spike SSD I/O on the next run.
- After restarts, prime the cache by issuing low-limit `PAIR_SCAN ctx:` passes (with cursors) or
  scripted `READ` loops over the namespaces the trainer/decoder will rely on. This shifts the
  initial churn into RAM and prevents a fresh workload from hammering the same disk sectors.
- When profiling raw disk I/O or working on RAM-starved systems, disable the cache by exporting
  `CHEETAH_PAYLOAD_CACHE_ENTRIES=0` (or the MB/bytes variants). Re-enable it immediately afterward so
  normal workloads keep leveraging RAM instead of falling back to repeated SSD seeks.

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
- Use `engine.iter_hot_context_hashes()` for namespace sweeps and
  `engine.context_relativism([...])` when you need probabilistic trie traversals with deterministic
  ordering. Both use `PAIR_SCAN` under the hood so traversals never touch SQL.

### DB Adapters

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
