# AI Reference

Fast-access knowledge base for CUDAway. Update this alongside `README.md` whenever code or docs
change so the next agent inherits the latest context.

## Collaboration Rules

- Mirror meaningful changes into `README.md`, `AI_REFERENCE.md`, and (if a study) `studies/*`.
- Read and record research results under `studies/` and cross-link them here so we avoid re-running the same
  investigations.
- Work incrementally, document breaking changes, and run the available build/test commands before
  yielding control.
- Always launch long-running services and workloads (smoke-train/benchmark runs, CI smoke tests, etc.) inside `screen` sessions: verify at the start of every screen invocation that no previous sessions are lingering, monitor the session output in real time, and attach explicit timeouts (≤30 minutes by default, ≤1 hour only if justified ahead of time) so stuck/error loops do not block the next agent. When the WSL image cannot keep `screen` alive (missing setuid bit), fall back to `tmux` with the exact same discipline and log the substitution in your notes/runbook. Service-specific tmux/watcher notes live in `cheetah-db/AI_REFERENCE.md`.

### Next Steps

`NEXT_STEPS.md` is now the single backlog source; consult it for active tooling tasks and update that file first before mirroring high-level context here. Update it with your new objectives, remove already completed and no more important steps to remember.

### Working With cheetah-db

Cheetah-specific operational steps and directives now live in `cheetah-db/AI_REFERENCE.md`. Read that file before launching the Go service, touching cheetah namespaces, or editing env vars such as `DBSLM_CHEETAH_*`.

- Trainer `--reset` now shrinks the cheetah namespace scan page size whenever `PAIR_SCAN` stalls and bumps the TCP idle-grace window to `max(DBSLM_CHEETAH_TIMEOUT_SECONDS * 180, 60)` seconds (override via `DBSLM_CHEETAH_IDLE_GRACE_SECONDS`, clamp via `DBSLM_CHEETAH_IDLE_GRACE_CAP_SECONDS`). Fresh databases therefore stop flooding the console with `cheetah response timed out after 30.0s of inactivity`, and slow disks can be accommodated by simply raising the timeout or idle-grace fields. When supported, `--reset` first issues `RESET_DB <DBSLM_CHEETAH_DATABASE>` to delete the entire cheetah namespace in one shot, then falls back to `PAIR_PURGE` (and finally the incremental scan loop) when older binaries lack the command.
- The hot-path adapter now queues reducers via `PAIR_REDUCE_ASYNC` and polls `PAIR_REDUCE_FETCH`, so the TCP socket never sits idle for minutes while cheetah walks slow namespaces. Tweak `CHEETAH_REDUCE_ASYNC` (disable to fall back to synchronous reducers) plus `CHEETAH_REDUCE_POLL_INTERVAL_SECONDS` to adjust the keep-alive cadence; synchronous fallbacks still honor the idle-grace clamp. Progress lines (state, % complete, and completed/total counts) are emitted roughly every 30s so long-running reducers remain visible in the trainer log.
- cheetah-db now keeps a bounded number of open pair-table handles. Idle trie nodes close their file descriptors and re-open automatically when accessed so long ingest runs no longer trip `too many open files`. Override the cap via `CHEETAH_MAX_PAIR_TABLES` (defaults to the detected `RLIMIT_NOFILE` minus a safety margin) when running on hosts with larger limits or very dense namespaces.
- Pair trie tables now ride on a managed I/O layer that caches 4 KiB sectors in RAM, queues dirty sectors through a background flusher, and only touches disk when necessary. Concurrent writers therefore hit in-memory buffers while the queue drains in the background, which both respects the descriptor cap and slashes SSD churn during heavy ingest.
- The managed file cache now listens to `ResourceMonitor` memory pressure and aggressively flushes/evicts idle sectors. Dirty pages are forced to disk (default 30s after last access, hard cap 5 minutes), freed immediately after the write completes, and low-usage sectors are culled whenever RAM pressure crosses ~90% (tunable via `CHEETAH_CACHE_*`). Set `CHEETAH_CACHE_IDLE_SECONDS`, `CHEETAH_CACHE_FORCE_SECONDS`, `CHEETAH_CACHE_SWEEP_SECONDS`, `CHEETAH_CACHE_STATS_SECONDS`, `CHEETAH_CACHE_PRESSURE_HIGH/LOW`, or the read/write weight knobs to bias which sectors survive.
- The Python hot-path adapter now retries failed `PAIR_SET` registrations (default 4 attempts) and verifies the mapping with `PAIR_GET` before giving up. Tune the behavior via `CHEETAH_PAIR_REGISTER_ATTEMPTS` and `CHEETAH_PAIR_REGISTER_BACKOFF_SECONDS` when mirroring namespaces over high-latency links.
- `src/train.py --cheetah-token-progress-interval` controls how often long-running cheetah prediction-table training emits progress logs (default 60s). Use this alongside the existing `--cheetah-token-*` knobs so evaluation cycles no longer appear "hung" while context matrices finish training.

## Prompt Tag Guardrails (High Priority)

- `DBSLMEngine.register_prompt_tags()` (`src/db_slm/pipeline.py`) now seeds the built-in structural tokens (`|INSTRUCTION|:`, `|USER|:`, `|RESPONSE|:`, `|CONTEXT|:`, etc.), appends every dataset-specific tag in discovery order, and forwards that enumeration to both the tokenizer and `ContextWindowEmbeddingManager`. Always call `collect_prompt_tag_tokens()` before ingest/eval so the global enumerator stays authoritative.
- Decoder sampling (`src/db_slm/pipeline.py` + `src/db_slm/decoder.py`) hard-bans those tokens and scans the first ~160 characters of every candidate for alias strings such as `user:` or `|instruction|:`. If any scaffold tag appears, the engine discards the text and re-runs decoding with a new RNG up to three times, mirroring how `|END|` retries work for too-short outputs.
- Prompt-tag bans and evaluation detection now normalize case when `DBSLM_TOKENIZER_LOWERCASE=1`, preventing lowercased tags (e.g., `|response|:`) from leaking into generations or slipping past retry gates.
- Context windows (`src/db_slm/context_window_embeddings.py`) now store the running mean/variance of each tag enumerator per dimension. These tag-aware weights flow into `ContextDimensionTracker`, so predicting a tag that belongs to a different prompt segment immediately increases the presence/frequency penalty even before the string-level guard activates.
- Level 3 `ContextSummary` payloads now stay internal to decoding. `DBSLMEngine.respond()` still feeds the summary into the rolling context and context-window bias text, but the synthesized `|CONTEXT|:` line is never prepended to user-visible generations, keeping datasets in full control of which tags reach the prompt/response stream.

## Codebase State

- `src/db_slm` now mirrors the v2 DB-SLM spec. Level 1 owns the vocabulary, tokenizer (regex by
  default or Hugging Face `tokenizers` when configured), hashed
  context registry, Modified Kneser–Ney smoother, quantized probability tables, and Top-K cache.
  Level 2 combines episodic logging, concept-ready correction digests, pointer-sentinel session
  caches, and token-level bias plumbing. Level 3 provides concept dictionaries, template
  verbalization, probability materialization, and conversation-scoped signals.
- `studies/DB_SLM_DATABASE_AND_ALGORITHMS.md` remains the authoritative schema/algorithm reference
  for the relational layout plus the KN materialization + decoding loops implemented in Python.
- `requirements.txt` now installs `sentence-transformers` (external embedding baseline),
  `language-tool-python` (grammar deltas for the quality gate), Hugging Face `tokenizers` (regex
  replacement backend), plus spaCy **and** Stanza so at least one dependency-parser stack is ready for
  the new training annotations. Optional GPU acceleration is auto-detected via PyTorch when present.
- `src/train.py` passes every JSON/NDJSON prompt/response through the dependency parser (preferring
  spaCy via `DBSLM_SPACY_MODEL`, falling back to Stanza via `DBSLM_DEP_LANG`/`DBSLM_STANZA_PROCESSORS`)
  and appends a `DependencyLayer: {...}` JSON blob to the staged corpus text. The blob enumerates the
  arcs plus a strong-token bucket list (subjects, objects, actions, modifiers, etc.) so the trainer
  and quality gate can weight those terms without growing the n-gram order. When neither backend is
  installed we emit a single warning and continue without the metadata.
- `src/train.py` parallelizes corpus staging (JSON chunking, dependency parsing, hold-out selection)
  via the new `--prep-workers` (default: `max(1, cpu_count-1)`) and `--prep-prefetch` options so the
  ingestion loop can stay CPU-bound on the DB/Smoother while workers feed ready-to-train chunks.
  Std-in payloads are still staged synchronously to avoid buffering surprises.
- Dataset configs now honor `context_fields[].placement` so structured metadata (for example the
  GPTeacher `instruction` column) can be injected ahead of the user prompt as `|INSTRUCTION|: ...`.
  `DatasetConfig.compose_prompt()` mirrors that preface for evaluation prompts, guaranteeing that
  training chunks and held-out probes share the same `|INSTRUCTION|` + `|USER|` framing. The
  inference CLI exposes `--instruction`, `--instruction-label`, and `--user-label` so `run.py`
  sessions can generate prompts with the identical scaffolding.
- Evaluation probes reuse the stored dependency layers to compute `strong_token_overlap` and
  `dependency_arc_overlap` metrics for each generation. Both are logged next to ROUGE/perplexity and
  folded into the metrics export so we can tell whether the decoder is preserving grammatical
  structure vs. just matching surface tokens.
- Trainer start-up now hard-fails when `language_tool_python` or a local `java` runtime are missing,
  so we no longer discover hours later that grammar metrics could not run. Install Java and
  `language-tool-python` before invoking `src/train.py`.
- Every staged response line now includes a trailing `|END|` tag. The decoder strips this token
  before responses hit the REPL/logs, and `EvalLogWriter` records a `cycle_reference` entry with the
  first full reference/generated sentence for each evaluation batch so the JSON timeline always
  captures a non-truncated exemplar for the cycle.
- Prompts handed to the decoder now always finish with the dataset-provided response label via
  `db_slm.prompt_tags.ensure_response_prompt_tag()`. `train.py`, `load_eval_dataset()`, and
  `run_inference_records()` call the helper before decoding, and `run.py` exposes
  `--response-label` for interactive sessions. This is a high-priority invariant: without the
  terminal `|RESPONSE|:` (or override) the model continues the `|USER|:` frame instead of producing a
  reply, corrupting both training chunks and eval probes.
- Dataset configs now advertise every `|TAG|:` prefix (prompt/response labels plus any context
  `canonical_tag` entry such as `|CTX|`), and `train.py` registers those sequences as atomic tokens
  so the regex tokenizer never splits them into stray pipes/colons. That guarantees stable vocab IDs
  for tags like `|USER|:` / `|INSTRUCTION|:` and lets each corpus opt-in to canonical context headers
  by setting `canonical_tag`; datasets that skip the field no longer receive automatic `|CTX|:`
  prefixes.
- Training runs with `--ngram-order` >= 5 now merge repeated token runs into composite tokens by default
  (`--merge-max-tokens 5`). Spans below the average frequency of all candidate spans are discarded,
  and runs dominated by high-frequency tokens are down-weighted so generic phrases are less likely
  to merge. The merge config is persisted in metadata (SQLite + cheetah) so inference reuses the
  same rules; set `--merge-max-tokens 0` to disable.
- `src/train.py` streams corpora into the SQLite store, triggering KN rebuilds + Top-K refreshes per
  ingest; `src/run.py` exposes the concept-aware REPL that performs Level 3 → Level 1 decoding with
  cache/bias adjustments. `run.py` now spawns a child decoder process (spawn context) so REPL input
  stays responsive on multi-core hosts; `:history` and the new `:status` alias proxy through the
  worker to fetch conversation windows/dim metadata.
- Long ingest phases now emit stage-aware progress lines (vocab, each n-gram order, smoothing) so
  large JSON chunks no longer look frozen; the logs include approximate line counts to show where
  the trainer is spending time.
- Added `src/log_helpers.log`, wired it through `src/train.py`, `src/run.py`, and `src/db_slm` helpers, and now every trainer/decoder line (including telemetry emitted by `scripts/smoke_train.py`) is prefixed with `+[seconds_since_start]`; backend-specific latency mirrors plus the tmux helpers are detailed inside `cheetah-db/AI_REFERENCE.md`.
- Realtime resource telemetry now flows through `src/helpers/resource_monitor.py`: during ingest profiles and every evaluation probe we record CPU %, RSS deltas, thread counts, and disk I/O (leveraging psutil with `resource` fallbacks) and push those samples both to the console and to the metrics export JSON.
- `CheetahHotPathAdapter` now spins up a dedicated cheetah-db TCP client per thread (with a shared factory + warm connection) so ingest, evaluation, and background workers can exercise true multi-core concurrency without funneling through a single socket; custom clients can still be injected for tests, but the default path uses the thread-local pool.
- `src/db_slm/sentence_parts.py` feeds `DBSLMEngine.train_from_text()` with punctuation-aware
  segments, embedding signatures, and context keyword tokens so Level 1 learns efficient splits in
  real time. Dataset metadata is now declared via `datasets/<name>.config.json`, which lets the
  loader emit canonical `|CTX|:<token>:<value>` lines whenever a context field sets `canonical_tag`; the sentence-part embedder lifts those tags
  into headers and supplements them with `|CTX_KEY|` keywords derived solely from the segment text +
  embedder energy (the `_EMOTION_WORDS` allowlist is gone). Configure the embedding backbone with
  `DBSLM_EMBEDDER_MODEL`, or force hashed-only, offline guidance (no Hugging Face downloads) via
  `DBSLM_EMBEDDER_OFFLINE=1`.
- Context dimensions now double as MiniLM-driven window embeddings. When you pass spans such as
  `--context-dimensions 6,12,24`, the trainer samples that many words per dimension (stride → 50%),
  embeds the windows with `all-MiniLM-L6-v2`, and keeps running prototypes per dimension inside both
  SQLite metadata and the cheetah hot-path namespace. During inference the decoder reuses those
  prototypes to scale the per-dimension presence/frequency penalties via cosine similarity, so the
  learned multi-scale contexts influence sampling without needing per-token embedding calls. The
  same context matrices now append dimension-level summary/fusion vectors so cheetah prediction
  tables see a hidden-layer style projection that differentiates short vs. long window signals, and
  extra fused tiers are added automatically when dimension summaries diverge; `--context-window-depth`
  biases how deep those tiers run (default now deeper). Training window sampling now adapts to token
  volume (with `--context-window-train-windows` acting as the cap), and prototype counts gate how deep
  those extra tiers run so early training stays shallow while richer corpora unlock more depth.
  Presets `default`, `deep`, and `shallow` are now accepted for `--context-dimensions` (default spans
  1-2,3-5,6-10,11-18; deep adds 19-31). cheetah-db now deepens every context matrix with derived
  mean/variance/contrast/interaction layers that scale with context diversity before prediction
  training/querying; toggle via `CHEETAH_PREDICT_DEEPEN=0` if needed. `train.py` also exposes
  `--context-window-train-windows`, `--context-window-infer-windows`, and `--context-window-stride-ratio`
  to control how densely those windows are sampled.
- `src/train.py` now persists a fast-resume state under `var/train_resume.json` and automatically
  resumes the last interrupted ingest when invoked with no arguments, skipping completed chunks.
- Future idea: promote each dimension to a small codebook (k-means per window size) and expose a CLI
  inspector so we can audit the learned prototypes or pin certain domains (code, poetry, etc.)
  before wiring them into penalty scaling.
- Tokenization now supports a Hugging Face-backed backend: set
  `DBSLM_TOKENIZER_BACKEND=huggingface`, point `DBSLM_TOKENIZER_JSON` at a tokenizer.json
  (usually exported from `tokenizers` or HF Hub), and optionally disable lower-casing with
  `DBSLM_TOKENIZER_LOWERCASE=0`. Missing packages or files trigger a logged warning and fall back to
  the legacy regex splitter so training/evals always proceed.
- Evaluation probes now request at least 20 generated words (scaling up toward the reference length)
  via a response backstop so lexical / ROUGE / perplexity logs never drop a row due to blank or
  truncated generations.
- `run_inference_records()` now retries with a fresh RNG seed whenever the decoder emits a prompt tag
  (|USER|, |INSTRUCTION|, |RESPONSE|, etc.) so evaluation logs capture actual completions instead of
  scaffolding tokens; retries still consume the `_MAX_BATCH_ATTEMPTS` budget to keep probes bounded.
- Evaluation now tracks cross-sample repetition via `helpers/char_tree_similarity.py`, exposes
  `char_repeat_max/avg` metrics, feeds them into `QualityGate`, and automatically re-queues variants
  with stronger presence/frequency penalties whenever the generated text matches earlier prompts too
  closely.
- Level 2 metadata now rides the same channel: `ConversationMemory` writes stats to
  `meta:l2:stats:<conversation_id>`, correction digests to `meta:l2:corr:<conversation_id>`, and
  bias presets to `meta:l2:bias:<conversation_id|__global__>`. Decoder/cache components consult those
  JSON blobs first so a restarted trainer no longer needs warm-up SQL reads before issuing concept or
  bias-aware generations. Metadata helpers now strip duplicate `meta:` prefixes and fall back to
  legacy `meta:meta:l2:*` entries so existing mirrors stay readable while new writes remain canonical.
- `src/train.py` now exposes `--profile-ingest` for RSS/latency logging and prints lexical overlap,
  ROUGE-L, plus generated/reference perplexity in every evaluation probe so we can quantify gains
  during long streaming ingests.
- `src/train.py` now accepts `--decoder-presence-penalty` and `--decoder-frequency-penalty` so repeat
  penalty grids can run directly from the CLI; overrides propagate to periodic + hold-out probes and
  are recorded inside the metrics metadata for downstream comparisons. These knobs only influence
  evaluation decoding—training statistics still reflect the raw corpus, so use higher penalties plus
  richer context dimensions when repetition creeps into probe outputs.
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
- Set `DEVICE=cuda` or `DEVICE=mps` before invoking `src/train.py` to run the sentence-transformer
  embedder plus the CoLA acceptability head on that accelerator whenever PyTorch reports it as
  available. Invalid requests log a single notice and the trainer falls back to CPU automatically.
- Evaluation probes now emit structural-diversity metrics (`structure_variety`, `common_token_penalty`,
  `token_group_share`, `top_token_share`, opener diversity, punctuation balance) that explicitly devalue templated
  responses. `QualityGate` ingests the same metrics so over-repeated openings or punctuation abuse
  get queued for retraining, and both periodic and chunk hold-out probes always run at least two
  samples (topped up from the rolling pool when needed). `SentenceQualityScorer` now scales
  `quality_score` down whenever `token_group_share` exceeds 0.30 so repetition spikes show up in the
  aggregate means.
- Adversarial prediction fixes now ride alongside evaluation. When a probe is flagged or its
  `quality_score` dips below `--cheetah-adversarial-threshold`, the trainer derives a context matrix
  from the prompt + metadata, reinforces the reference tokens, and down-weights the generated tokens
  with a single `PREDICT_TRAIN` call that carries `negatives=`. Tune via
  `--disable-cheetah-adversarial-train`, `--cheetah-adversarial-max-negatives`, and
  `--cheetah-adversarial-learning-rate` (defaults to 60% of the main cheetah-token rate) so bad
  generations immediately correct the prediction table instead of waiting for a full retrain.
- Prediction tables now persist normalized window hints from every training/adversarial context. The
  Go reducers blend those stored hints with any caller-supplied windows (or fall back to the hints
  when none are provided), exposing hidden correlations in context matrices without extra CLI
  arguments.
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
- `scripts/run_paraphraser_regression.py` consumes `studies/paraphraser_regression.jsonl` to ensure
  multi-turn corrective threads, structural tags, and ordinary prompts all trigger the expected
  paraphraser behavior.
- Inserts/edits now seed the cache and deletes invalidate their slots, so you can chain
  ingest → reducer → decoder without reopening the same offsets. Keep the server alive between
  pipeline stages to retain the warm cache; restarting the process will cold-start the cache and
  briefly spike SSD I/O on the next run.
- After restarts, prime the cache by issuing low-limit `PAIR_SCAN ctx:` passes (with cursors) or
  scripted `READ` loops over the namespaces the trainer/decoder will rely on. This shifts the
  initial churn into RAM and prevents a fresh workload from hammering the same disk sectors.
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
- Dataset-specific schema is declared beside each corpus as `datasets/<name>.config.json`
  (see `datasets/emotion_data.config.json`). The loader uses it to map prompt/response fields,
  expose optional context tokens, and emit canonical headers such as `|CTX|:<token>:<value>` whenever a context field sets `canonical_tag`, so no code change is required when swapping corpora.
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
- The telemetry thread now exposes `--queue-drain-threshold`, `--queue-drain-cooldown`,
  `--queue-drain-{script,metrics-dir,benchmarks}`, and `--disable-auto-queue-drain`. Leaving the
  automation enabled means queue overflows trigger `scripts/drain_queue.py` automatically and the
  resulting metrics block is mirrored into `studies/BENCHMARKS.md` without manual editing.
- Use `--scenarios a,b` or `SMOKE_SCENARIOS=a,b` to run a subset, `--resume-from name` to skip ahead,
  and `--dry-run` to print the commands while still updating `benchmarks.json` for planning.

- Use `engine.iter_hot_context_hashes()` for namespace sweeps and
  `engine.context_relativism([...])` when you need probabilistic trie traversals with deterministic
  ordering. Both use `PAIR_SCAN` under the hood so traversals never touch SQL.

### DB Adapters
