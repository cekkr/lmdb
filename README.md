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
  pip install -r requirements.txt  # currently empty but preserves the workflow
  ```
- The CLI utilities default to storing everything under `var/db_slm.sqlite3`. Feel free to point them
  at any other SQLite path or even `:memory:` when using the programmatic API.
- Even though `.env` exposes MariaDB credentials, the reference CLI still targets SQLite until we run
  the schema migration step. Seeing no MySQL tables during local training is therefore expected.

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
- `--stdin`: Stream additional ad-hoc text directly from `STDIN`, e.g. `cat notes.txt | python src/train.py --stdin`.
- `--encoding`: Override the default UTF-8 reader if your corpus uses a different encoding.

The script reports per-file token counts plus the aggregate number of stored N-grams. If the provided
corpora are shorter than the N-gram order they are automatically skipped.

Validation helpers shipped with `train.py` make it easier to work with huge NDJSON datasets such as
`datasets/emotion_data.json`:

- `--json-chunk-size`: stream JSON/NDJSON rows in fixed-size batches so the process never loads the
  full file into memory.
- `--max-json-lines`: cap the number of JSON rows read per file when you only need a quick smoke test.
- `--eval-interval`, `--eval-samples`, `--eval-dataset`, `--eval-pool-size`: enable periodic
  inference probes during training to log qualitative progress without leaving the script.
- `--chunk-eval-percent`: when ingesting JSON/NDJSON corpora, reserve this percentage of every chunk
  as hold-out prompts/responses. Those samples skip training entirely and are immediately replayed
  through the inference path so you can spot regressions tied to the latest data instead of a fixed
  seed set.
- `--profile-ingest`: print per-corpus latency and resident-set-size metrics so you can keep raising
  `--json-chunk-size` / `--max-json-lines` until memory pressure kicks in. On a 16 GB laptop, chunks
  of ~2,000 rows (~4 MB) kept RSS under 2.5 GB; bigger batches introduced GC pauses, so we documented
  that tipping point directly in the training logs.

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
verbatim echo.

## Smoke Testing

A minimal regression path is wired to `make smoke-train`. It performs a capped ingest (400 NDJSON
lines), runs the periodic evaluation probes with the new metrics, and issues a single REPL-style
prompt so CI or local developers can confirm the full pipeline still works:

```bash
make smoke-train
```

Use `make clean-smoke` to remove the temporary database when you're done.

## MariaDB Migration

Once the SQLite validation database looks healthy, run:

```bash
# Generate a portable SQL script
python scripts/migrate_sqlite_to_mariadb.py --sqlite var/db_slm.sqlite3 --output var/mariadb.sql

# Optionally push it straight into MariaDB (requires mysql-connector-python)
python scripts/migrate_sqlite_to_mariadb.py --apply --drop-existing
```

The migration utility introspects the SQLite schema/tables, emits MariaDB-compatible DDL +
INSERTs, and (when `--apply` is selected) replays them against the credentials from `.env`. This
keeps the production backend in sync with the SQLite dev instance without manually rewriting schema
definitions.
