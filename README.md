# lmdb
[Experimental] Database centric LLM

## Overview

This repository explores the production-grade database-native statistical language model (DB-SLM)
described in `studies/CONCEPT.md` and refined in
`studies/DB_SLM_DATABASE_AND_ALGORITHMS.md`. The Python implementation under `src/db_slm` now mirrors
the spec end-to-end:

- **Level 1 — Aria-style n-gram engine:** Byte-free (regex) tokenization backed by a relational
  vocabulary, hashed contexts, Modified Kneser–Ney smoothing, quantized log-prob tables, and a
  Top-K materialization path for fast sampling. All stats live inside SQLite tables that mirror the
  MariaDB layout.
- **Level 2 — Episodic memory + biasing:** Conversation logging, correction digests, logit-bias
  materialization, and pointer-sentinel session caches that feed the decoder without tensors.
- **Level 3 — Concept model:** Concept dictionaries, templates, and probability tables that can
  output multi-token spans before the Level 1 stitcher runs.

Training, inference, cache mixtures, and bias application all happen through SQL updates and lookups.

## Quick Start

```python
from db_slm import DBSLMEngine

engine = DBSLMEngine()
conversation_id = engine.start_conversation(user_id="demo")
print(engine.respond(conversation_id, "Remind me what we discussed."))
```

Use `train_from_text()` to ingest corpora. It automatically updates counts, rebuilds the KN
probabilities, and refreshes the Top-K cache so the decoder can read quantized distributions directly
from the database.

## Environment Setup

- Python 3.10+ is recommended. Create an isolated virtual environment if you plan to experiment:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- The CLI utilities default to storing everything under `var/db_slm.sqlite3`. Feel free to point them
  at any other SQLite path or even `:memory:` when using the programmatic API.
- Even though `.env` exposes MariaDB credentials, the reference CLI still targets SQLite until we run
  the schema migration step. Seeing no MySQL tables during local training is therefore expected.

### cheetah-db hot path (optional)

- `cheetah-db/` hosts the Go service that now acts as a low-latency mirror for Level 1 contexts.
  Build it with `bash cheetah-db/build.sh` and launch `./cheetah-db/cheetah-server` before
  running the Python tools.
- Set `DBSLM_BACKEND=cheetah-db` when you want the trainer/decoder to fetch Top-K slices from the
  Go engine instead of SQLite. When you only need a sidecar cache (SQLite remains canonical), leave
  the backend as `sqlite` and toggle `DBSLM_CHEETAH_MIRROR=1`.
- The `DBSLM_CHEETAH_HOST/PORT/DATABASE` variables point the adapter at the right instance; the
  default matches the server exposed by `cheetah-db/main.go`.
- During ingest the Python pipeline now streams new context metadata and Top-K probability slices
  into cheetah so the decoder can read them without re-querying SQLite, satisfying the first step of
  the adapter roadmap outlined in `cheetah-db/README.md`.

## Training CLI (`src/train.py`)

`train.py` is the canonical way to populate the Level 1 n-gram tables from plain-text corpora. Each
file is streamed into `DBSLMEngine.train_from_text()`, which updates hashed context counts, rebuilds
Modified Kneser–Ney statistics for every order up to the configured `--ngram-order`, refreshes
continuation counts, and re-materializes quantized probability tables plus the Top-K head cache.

```bash
python src/train.py \
  data/corpus.txt docs/*.txt \
  --db var/db_slm.sqlite3 \
  --ngram-order 3 \
  --recursive \
  --reset
```

Key options:

- `inputs`: One or more files or directories containing `.txt` files. Directories are scanned
  recursively when `--recursive` is supplied.
- `--db`: Destination SQLite file. The parent directory is created automatically. Use `--reset` when
  you want to erase the previous database before ingesting.
- `--ngram-order`: Controls the window length. Increase for more context, decrease for tiny corpora.
- `--context-dimensions "<ranges>"`: Extends the repeat penalty across grouped token spans. Accepts
  comma-separated ranges such as `1-2,3-5` (the default) to cover word- and sentence-length windows,
  a single integer (e.g., `4`), or `off`/`none` to disable the extra grouping penalty. The selection
  is persisted inside `tbl_metadata` so later CLI invocations inherit the same spans.
- `--stdin`: Stream additional ad-hoc text directly from `STDIN`, e.g. `cat notes.txt | python src/train.py --stdin`.
- `--encoding`: Override the default UTF-8 reader if your corpus uses a different encoding.

The script reports per-file token counts plus the aggregate number of stored N-grams. If the provided
corpora are shorter than the N-gram order they are automatically skipped.

Validation helpers shipped with `train.py` make it easier to work with huge NDJSON datasets such as
`datasets/emotion_data.json`:

- `--json-chunk-size`: stream JSON/NDJSON rows in fixed-size batches so the process never loads the
  full file into memory.
- `--max-json-lines`: cap the number of JSON rows read per file when you only need a quick smoke test.
- **Evaluation controls (all optional, 0 disables the feature):**
  - `--eval-interval <tokens>`: trigger periodic probes every N ingested tokens (default `0`, meaning
    disabled). Context-dimension runs automatically emit two generations per prompt to compare span
    penalties in real time.
  - `--eval-samples <count>`: number of held-out prompts per probe (minimum 2, default `3`).
  - `--eval-dataset <path>`: NDJSON file with `prompt`/`response` pairs; defaults to
    `DBSLM_DATASET_PATH` from `.env`.
  - `--eval-pool-size <count>`: cap (or unset for unlimited) on how many records remain in the rolling
    evaluation buffer.
  - `--chunk-eval-percent <0-100>`: reserve this percent of every JSON chunk as an immediate
    hold-out set; they run through the inference stack before the chunk trains and refresh the pool.
- `--profile-ingest`: print per-corpus latency and resident-set-size metrics so you can keep raising
  `--json-chunk-size` / `--max-json-lines` until memory pressure kicks in. On a 16 GB laptop, chunks
  of ~2,000 rows (~4 MB) kept RSS under 2.5 GB; bigger batches introduced GC pauses, so we documented
  that tipping point directly in the training logs.
- `--metrics-export <path>`: write the rolling ROUGE/perplexity timeline plus profiling samples to a
  JSON artifact. When omitted, `train.py` automatically drops `var/eval_logs/train-<timestamp>.json`
  so you can compare runs without scraping stdout. Use `--metrics-export -` to disable the feed.
- **Penalty overrides (per-probe sampling tweaks):**
  - `--decoder-presence-penalty <float>=0+`: add a one-time penalty whenever a token (or grouped span)
    has already appeared in the current generation. Typical sweep range is `0.0-0.4`.
  - `--decoder-frequency-penalty <float>=0+`: scales with how many times the token/span has appeared.
    Values between `0.0` and `0.2` generally work best for conversational corpora.

Large batches can take a while to finish a single call to `train_from_text()`, so the trainer now
prints progress lines for the vocabulary pass, every n-gram order, and the KN rebuilds. The logs
include an approximate row number so you can tell which part of the chunk is currently in flight
instead of staring at a silent terminal.

Example (quick validation run that ingests only 200 lines and probes the decoder every ~2k tokens):

```bash
python3 src/train.py datasets/emotion_data.json \
  --db var/db_slm.sqlite3 \
  --reset \
  --json-chunk-size 100 \
  --max-json-lines 200 \
  --eval-interval 2000 \
  --eval-samples 2 \
  --eval-pool-size 20
```

### Adaptive Tokenization + Emotional Embeddings

`DBSLMEngine` now performs a realtime corpus scan before every ingest to discover the most productive
punctuation splits, slice long responses into manageable segments, and tag those fragments with
device-aware embeddings from `sentence-transformers` (defaults to `all-MiniLM-L6-v2`, configurable
via `DBSLM_EMBEDDER_MODEL`). Each segment is annotated with a compact embedding signature plus an
`|EMO_KEY|` list that surfaces emotional keywords derived from both the dataset metadata (the
`Emotion:` header) and the embedding energy. These annotations are injected ahead of the regex
tokenizer, ensuring the vocabulary learns explicit emotional cues and higher quality boundary splits
even while the underlying Level 1 tables remain purely relational. When the optional dependency is
missing, the pipeline falls back to deterministic hashed vectors so tokenization still benefits from
the dataset profiler.

### Automatic MariaDB Flush for Cold Contexts

Long-running SQLite sessions can now hand cold Level 1 contexts to MariaDB automatically. Set the
threshold via `.env` (`DBSLM_SQLITE_FLUSH_THRESHOLD_MB`, defaults to 1024 MB). When the on-disk file
exceeds that value, the new `ColdStorageFlusher` ships up to `DBSLM_SQLITE_FLUSH_BATCH` low-usage
contexts—ranked by `hot_rank`, `total_count`, and `updated_at`—into the MariaDB tables (creating them
if they do not already exist) and deletes the same rows from SQLite. This keeps local training runs
from exhausting RAM while preserving a lossless copy of the rarely used statistics in the downstream
database for later replay. Install `mysql-connector-python` (already included in
`requirements.txt`) and provide valid MariaDB credentials in `.env` to activate the flush path.

### Training-Time Metrics

Every evaluation probe now prints lexical-overlap, ROUGE-L, and a perplexity stub for both the
generated response and the held-out reference. This lets you confirm that quantitative scores improve
while you experiment with chunk sizes or smoothing tweaks, even when the qualitative samples look
similar. When `--chunk-eval-percent` is supplied, the same metric stack runs immediately after each
chunk ingest using its freshly carved-out hold-outs, giving you a rolling measure that tracks the
latest corpus segment instead of relying solely on the static evaluation dataset. Probes start a
fresh conversation with the low-resource seeding helper disabled, so the reported generations will
reflect the newly trained n-gram tables instead of the caretaker seed dialog you may see in
interactive `run.py` sessions on tiny corpora.

In addition to the per-sample logs, `train.py` now prints run-level averages (lexical overlap,
ROUGE-L, generated/reference perplexity) after every probe and mirrors the raw samples into
`var/eval_logs/*.json`. The feed includes hold-out probes, periodic evaluation sets, and optional
profiling records so you can diff long runs or export the JSON into your own dashboards. Point
`--metrics-export` at a custom path when you need to archive the file elsewhere.

Evaluation probes also call the new sentence-quality stack: LanguageTool for grammar deltas,
`textattack/roberta-base-CoLA` for semantic acceptability, and the shared sentence-transformer
embedder for similarity/novelty scores. Those numbers are appended to the JSON timeline alongside
lexical/ROUGE/perplexity values, so you can catch regressions that only manifest as grammatical
errors or semantic drift. Because `emotion_data.json` responses average ~347 words, the evaluator
derives `min_response_words` from the reference length (capped at 512 words) to ensure the logged
`|RESPONSE|` frame actually reaches the substantive part of the answer instead of truncating after
128 words. When a sample is flagged for retraining it now re-enters the current batch at a random
position (up to two total attempts) before being scheduled for future probes, so the decoder gets a
fresh shot without holding up the rest of the evaluation.

Supplying `--decoder-presence-penalty` or `--decoder-frequency-penalty` only affects the inference
path used by those probes (including chunk hold-outs); training statistics stay unchanged. The chosen
values flow into the `DecoderConfig` passed to `issue_prompt()` and are emitted in the metadata block
inside `var/eval_logs/*.json`, so repeat-penalty sweeps can be compared later without scraping the
console logs.

Low-quality generations (grammar errors ≥ 3, CoLA < 0.45, semantic similarity < 0.55, or a >40%
length mismatch) are streamed into `DBSLM_QUALITY_QUEUE_PATH` (defaults to
`var/eval_logs/quality_retrain_queue.jsonl`). This “retrain queue” doubles as a regression fixture:
drop the file back into `train.py` as an evaluation dataset and the weakest samples receive targeted
attention during the next ingest. Heavy grammar/semantic scoring only runs when the adaptive CPU
guard detects spare headroom, so long streaming ingests do not pay a latency penalty on saturated
laptops.

## Inference CLI (`src/run.py`)

`run.py` spins up a conversational REPL backed by the database produced during training. The loop
invokes the full decoding pipeline: Level 3 concept prediction (with signal overrides), Level 2 bias
and cache adjustments, and Level 1 top-p decoding with quantized probabilities.

Interactive session:

```bash
python src/run.py --db var/db_slm.sqlite3
[run] Using conversation: 6f5080d1-...
you> summarize our discussion
assistant> Based on our recent exchange: ...
you> :history   # prints the Level 2 context window
you> :exit
```

Single-shot inference:

```bash
python src/run.py --db var/db_slm.sqlite3 --prompt "Remind me what we covered."
```

After a limited validation run (like the example above) you can immediately inspect the model with:

```bash
python3 src/run.py --db var/db_slm.sqlite3 --prompt "Summarize the role of empathy in leadership."
```

Useful flags:

- `--conversation`: Resume a previous conversation ID logged in `tbl_l2_conversations`. Omit to start a
  fresh session (the ID is printed on startup).
- `--user` / `--agent`: Override the identifiers written into Level 2 so you can distinguish sessions.
- `--max-turns`: Cap the number of turns before the REPL exits automatically. Handy for scripted demos.
- Commands inside the REPL: `:history` to print the current Level 2 context window, `:exit`/`:quit` (or
  `Ctrl+D`) to leave.

Because the CLI uses the exact same engine object, anything logged via `run.py` is immediately
available to downstream tooling (correction logging, concept payload providers, etc.).

Small validation runs sometimes overfit; the engine now seeds each conversation with short caretaker
exchanges and paraphrases overly similar replies so you still get meaningful summaries instead of a
verbatim echo. Multi-turn prompts and corrective instructions are explicitly guarded, so the
paraphraser never rewrites structured guidance or follow-up directions.

`scripts/run_paraphraser_regression.py` exercises those guard rails against
`studies/paraphraser_regression.jsonl`, which mixes multi-turn corrective threads, structural tags,
and plain prompts that should still be rewritten. Wire it into CI or run it locally whenever you
tweak `SimpleParaphraser`.

Training-time evaluations were further hardened so the decoder always produces at least 20 words,
even when the probabilistic backoff is uncertain. The new response backstop adds transparent filler
sentences referencing the prompt keywords so ROUGE/perplexity measurements never silently drop rows.

## Smoke Testing

A minimal regression path is wired to `make smoke-train`. It performs a capped ingest (400 NDJSON
lines), runs the periodic evaluation probes with the new metrics, and issues a single REPL-style
prompt so CI or local developers can confirm the full pipeline still works:

```bash
make smoke-train
```

Use `make clean-smoke` to remove the temporary database when you're done.

## SQLite vs. MariaDB Workflow

SQLite stays in the loop for day-to-day training and interactive inference because it behaves well on
laptops, WAL keeps ingest latency predictable, and the Level 1 counts fit comfortably inside a single
file. Once the corpus grows, the `ColdStorageFlusher` starts streaming rarely accessed contexts into
MariaDB so you get a lossless archival copy without ballooning local RSS. Downstream inference stacks
(or analytics jobs) can attach directly to MariaDB, while SQLite keeps the hot contexts close to the
trainer.

When you're ready to sync the two stores, run:

```bash
# Generate a portable SQL script for QA/backup
python scripts/migrate_sqlite_to_mariadb.py --sqlite var/db_slm.sqlite3 --output var/mariadb.sql

# Dry-run the incremental upsert against the MariaDB DSN from .env
python scripts/migrate_sqlite_to_mariadb.py --apply --incremental --dry-run

# Apply the same bundle for real (no rollback) once the dry-run looks clean
python scripts/migrate_sqlite_to_mariadb.py --apply --incremental
```

`--dry-run` wraps the entire MariaDB session inside a rollback-only transaction, so you can confirm
that upserts leave staging tables untouched before flipping the nightly job over. Use
`--drop-existing` only when you explicitly need a full rebuild; the default incremental mode mirrors
the SQLite counts via `INSERT ... ON DUPLICATE KEY UPDATE` so both backends stay logically identical.
