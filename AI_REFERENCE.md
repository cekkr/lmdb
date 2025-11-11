# AI Reference

Fast-access knowledge base for CUDAway. Update this alongside `README.md` whenever code or docs
change so the next agent inherits the latest context.

## Collaboration Rules

- Mirror meaningful changes into `README.md`, `AI_REFERENCE.md`, and (if a study) `studies/*`.
- Read and record research results under `studies/` and cross-link them here so we avoid re-running the same
  investigations.
- Work incrementally, document breaking changes, and run the available build/test commands before
  yielding control.

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
- `requirements.txt` now installs `sentence-transformers` (for the external embedding baseline) and
  `mysql-connector-python` (for automated cold-context flushing). Optional GPU acceleration is
  auto-detected via PyTorch when present.
- `src/train.py` streams corpora into the SQLite store, triggering KN rebuilds + Top-K refreshes per
  ingest; `src/run.py` exposes the concept-aware REPL that performs Level 3 → Level 1 decoding with
  cache/bias adjustments.
- Long ingest phases now emit stage-aware progress lines (vocab, each n-gram order, smoothing) so
  large JSON chunks no longer look frozen; the logs include approximate line counts to show where
  the trainer is spending time.
- `src/db_slm/sentence_parts.py` feeds `DBSLMEngine.train_from_text()` with punctuation-aware
  segments, embedding signatures, and emotion keyword tokens so Level 1 learns efficient splits in
  real time. Configure the embedding backbone with `DBSLM_EMBEDDER_MODEL`.
- Evaluation probes now request at least 20 generated words (scaling up toward the reference length)
  via a response backstop so lexical / ROUGE / perplexity logs never drop a row due to blank or
  truncated generations.
- `ColdStorageFlusher` (wired into `train.py`) monitors the SQLite file size and automatically
  migrates low-frequency contexts into MariaDB once `DBSLM_SQLITE_FLUSH_THRESHOLD_MB` is hit,
  removing the same rows locally to keep memory pressure in check.
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
- Latest smoke-train (2025-11-10, python3.11, `datasets/emotion_data.json` capped at 400 rows) logged
  882k tokens with avg quality 0.599 and structure_variety 0.317 (details in `studies/BENCHMARKS.md`);
  keep an eye on the 64% flagged rate--pool diversity or penalty tuning may be needed before next run.
  `QualityGate.common_token_ceiling` now defaults to 0.55 to drive the flagged rate below 45% once the
  token-group repetition penalty feeds into `quality_score`.
- Evaluation retries for flagged samples are now capped at two attempts per batch, with flagged rows
  re-queued into a random spot of the current probe before being eligible for up to three additional
  appearances in later random batches so probes cannot loop forever when the generator keeps
  emitting low-quality responses.
- The evaluator infers `min_response_words` from the reference length (capped at 512) so long-form
  corpora like `emotion_data.json` do not lose the substantive portion of the `|RESPONSE|` frame, and
  CPU-heavy quality scoring is gated behind the adaptive load monitor to avoid starving ingestion.
- Both `train.py` and `run.py` now rely on `db_slm.inference_shared.issue_prompt()` so scripted probes
  and the REPL reuse the same conversation bootstrapper.
- `scripts/migrate_sqlite_to_mariadb.py` converts the SQLite store into MariaDB-ready DDL + data and
  can optionally apply it directly using credentials from `.env`. `--incremental` performs
  `INSERT ... ON DUPLICATE KEY UPDATE` cycles so nightly refreshes no longer need to drop tables.
- `scripts/run_paraphraser_regression.py` consumes `studies/paraphraser_regression.jsonl` to ensure
  multi-turn corrective threads, structural tags, and ordinary prompts all trigger the expected
  paraphraser behavior.
- `Makefile` now includes `smoke-train`, a capped ingest + inference probe suitable for CI health
  checks.

## Operational Notes

### Streaming Ingest / Profiling

- Enable `--profile-ingest` whenever you push `--json-chunk-size` above 500 rows. The profiler logs
  per-corpus latency plus RSS deltas so you can note the tipping point in `NEXT_STEPS.md` (current
  guidance: stay under ~2.5 GB RSS and <5 s per chunk on 16 GB laptops).
- Let the held-out probes run with ROUGE/perplexity enabled so you can correlate throughput tweaks
  with quantitative gains instead of relying on overlap logs only.
- `datasets.md` now tracks basic stats for `emotion_data.json` (avg response 347 words, max 1,251) so
  chunk sizes, eval thresholds, and paraphraser guard rails stay grounded in the actual corpora.
- Prefer `make smoke-train` for regressions since it wires the capped ingest + inference probe into
  a single command and exercises the paraphraser path automatically.

### MariaDB Handoff

- Generate a SQL artifact after major SQLite validations:
  ```
  python scripts/migrate_sqlite_to_mariadb.py \
    --sqlite var/db_slm.sqlite3 \
    --output var/mariadb-release.sql
  ```
- Use `--apply --incremental` to upsert rows in place for nightly refreshes that should avoid table
  drops. Fall back to `--drop-existing` only when you explicitly need a clean rebuild.
- Install `mysql-connector-python` locally before invoking `--apply`; it replays schema + data and
  will drop the destination tables only when `--drop-existing` is explicitly set (announce that step
  before touching prod).
- `--dry-run` keeps the MariaDB session inside a rollback-only transaction so you can confirm
  incremental upserts leave staging tables untouched before flipping cron jobs over to the real run.
  Keep training on SQLite for locality; the `ColdStorageFlusher` and migration script keep MariaDB in
  sync for archival queries and downstream inference.

### DB Adapters

- `sqlite` remains the compatibility backend exposed through `DBSLMSettings.backend`, but it is now
  considered legacy. Keep it only for metadata bootstrapping while the Level 1 pipelines move over
  to cheetah; new work should avoid adding tables or features that would be SQLite-only and instead
  target the Go engine.
- `mariadb` (MySQL) served as the cold-storage target for `ColdStorageFlusher`. With cheetah
  streaming online the flusher is slated for removal, so treat the `DBSLM_MARIADB_*` knobs as
  deprecated and plan to delete them once cheetah owns archival duties.
- `cheetah-db` (see `cheetah-db/`) now doubles as the hot-path mirror for contexts and Top-K
  slices. Setting `DBSLM_BACKEND=cheetah-db` (or leaving the backend as `sqlite` and enabling
  `DBSLM_CHEETAH_MIRROR=1`) makes the trainer push every newly discovered context and MKNS Top-K
  bucket into the Go service via its TCP commands; the decoder then queries cheetah first and falls
  back to SQLite when a key is missing. As of this pass:
  - the trie now exposes a `PAIR_SCAN` streaming command so ordered byte slices can be pulled
    without touching SQLite, and the Python adapter exposes it via
    `HotPathAdapter.scan_namespace()/CheetahClient.pair_scan`;
  - server-side reducers/stats refreshers are still pending so MKNS rebuilds can stay inside
    cheetah;
  - brute-force sweep helpers are still required so Level 2/3 jobs can shard across cheetah files.
  Keep `NEXT_STEPS.md` updated with those gaps and record interoperability details in
  `cheetah-db/README.md` for future agents, since the roadmap now aims to delete the SQLite and
  MariaDB code paths once the reducers land.
