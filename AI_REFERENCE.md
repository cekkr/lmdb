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
- Evaluation probes reuse the stored dependency layers to compute `strong_token_overlap` and
  `dependency_arc_overlap` metrics for each generation. Both are logged next to ROUGE/perplexity and
  folded into the metrics export so we can tell whether the decoder is preserving grammatical
  structure vs. just matching surface tokens.
- `src/train.py` streams corpora into the SQLite store, triggering KN rebuilds + Top-K refreshes per
  ingest; `src/run.py` exposes the concept-aware REPL that performs Level 3 → Level 1 decoding with
  cache/bias adjustments.
- Long ingest phases now emit stage-aware progress lines (vocab, each n-gram order, smoothing) so
  large JSON chunks no longer look frozen; the logs include approximate line counts to show where
  the trainer is spending time.
- Added `src/log_helpers.log`, wired it through `src/train.py`, `src/run.py`, and `src/db_slm` helpers, and now every trainer/decoder line (including telemetry emitted by `scripts/smoke_train.py`) is prefixed with `+[seconds_since_start]`; backend-specific latency mirrors plus the tmux helpers are detailed inside `cheetah-db/AI_REFERENCE.md`.
- `src/db_slm/sentence_parts.py` feeds `DBSLMEngine.train_from_text()` with punctuation-aware
  segments, embedding signatures, and context keyword tokens so Level 1 learns efficient splits in
  real time. Dataset metadata is now declared via `datasets/<name>.config.json`, which lets the
  loader emit canonical `|CTX|:<token>:<value>` lines; the sentence-part embedder lifts those tags
  into headers and supplements them with `|CTX_KEY|` keywords derived solely from the segment text +
  embedder energy (the `_EMOTION_WORDS` allowlist is gone). Configure the embedding backbone with
  `DBSLM_EMBEDDER_MODEL`, or force hashed-only, offline guidance (no Hugging Face downloads) via
  `DBSLM_EMBEDDER_OFFLINE=1`.
- Tokenization now supports a Hugging Face-backed backend: set
  `DBSLM_TOKENIZER_BACKEND=huggingface`, point `DBSLM_TOKENIZER_JSON` at a tokenizer.json
  (usually exported from `tokenizers` or HF Hub), and optionally disable lower-casing with
  `DBSLM_TOKENIZER_LOWERCASE=0`. Missing packages or files trigger a logged warning and fall back to
  the legacy regex splitter so training/evals always proceed.
- Evaluation probes now request at least 20 generated words (scaling up toward the reference length)
  via a response backstop so lexical / ROUGE / perplexity logs never drop a row due to blank or
  truncated generations.
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
  expose optional context tokens, and emit canonical `|CTX|:<token>:<value>` headers so no code
  change is required when swapping corpora.
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
