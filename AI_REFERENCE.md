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
- `src/db_slm/sentence_parts.py` feeds `DBSLMEngine.train_from_text()` with punctuation-aware
  segments, embedding signatures, and emotion keyword tokens so Level 1 learns efficient splits in
  real time. Configure the embedding backbone with `DBSLM_EMBEDDER_MODEL`.
- Evaluation probes now request at least 20 generated words via a response backstop so lexical /
  ROUGE / perplexity logs never drop a row due to blank generations.
- `ColdStorageFlusher` (wired into `train.py`) monitors the SQLite file size and automatically
  migrates low-frequency contexts into MariaDB once `DBSLM_SQLITE_FLUSH_THRESHOLD_MB` is hit,
  removing the same rows locally to keep memory pressure in check.
- `src/train.py` now exposes `--profile-ingest` for RSS/latency logging and prints lexical overlap,
  ROUGE-L, plus generated/reference perplexity in every evaluation probe so we can quantify gains
  during long streaming ingests.
- `src/train.py` can reserve a slice of every JSON/NDJSON chunk for immediate evaluation via
  `--chunk-eval-percent`; those hold-out prompts/responses skip training and run through the same
  inference metrics the moment the chunk finishes ingesting, giving tighter feedback loops tied to
  the newest data.
- Training-time probes now spin up seedless conversations so low-resource scaffolding never masks
  evaluation outputs; if you still need the canned turns (e.g., via `run.py`) the helper stays
  enabled for interactive sessions only.
- `DBSLMEngine` still seeds low-resource conversations with two caretaker turns, but the paraphraser
  now uses length-aware thresholds and explicit guard rails so multi-turn prompts or corrective
  instructions are never rewritten while we avoid verbatim echoes.
- Evaluation summaries are written both to stdout and to structured JSON under
  `var/eval_logs/train-*.json`. Set `--metrics-export <path>` (or `-` to disable) to control the feed,
  which captures probe averages plus optional ingest profiling samples.
- Both `train.py` and `run.py` now rely on `db_slm.inference_shared.issue_prompt()` so scripted probes
  and the REPL reuse the same conversation bootstrapper.
- `scripts/migrate_sqlite_to_mariadb.py` converts the SQLite store into MariaDB-ready DDL + data and
  can optionally apply it directly using credentials from `.env`. `--incremental` performs
  `INSERT ... ON DUPLICATE KEY UPDATE` cycles so nightly refreshes no longer need to drop tables.
- `Makefile` now includes `smoke-train`, a capped ingest + inference probe suitable for CI health
  checks.

## Operational Notes

### Streaming Ingest / Profiling

- Enable `--profile-ingest` whenever you push `--json-chunk-size` above 500 rows. The profiler logs
  per-corpus latency plus RSS deltas so you can note the tipping point in `NEXT_STEPS.md` (current
  guidance: stay under ~2.5 GB RSS and <5 s per chunk on 16 GB laptops).
- Let the held-out probes run with ROUGE/perplexity enabled so you can correlate throughput tweaks
  with quantitative gains instead of relying on overlap logs only.
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
