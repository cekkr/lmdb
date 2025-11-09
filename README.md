# lmdb
[Experimental] Database centric LLM

## Overview

This repository explores the database-native statistical language model (DB-SLM) designed in
`studies/CONCEPT.md`. Instead of tensors, all generation stages are persisted as relational tables:

- **Level 1 — Statistical N-grams (Aria-like):** Lookup tables for token probabilities.
- **Level 2 — Stateful Memory (MyRocks + InnoDB):** Conversation logs and correctional RAG.
- **Level 3 — Conceptual Generation (InnoDB):** Concept dictionaries, templates, and selection logic.

See `studies/DB_SLM_DATABASE_AND_ALGORITHMS.md` for the storage layout plus the training and inference
playbooks that tie these levels together.

The first Python scaffolding for this design now lives under `src/db_slm`.

## Quick Start

```python
from db_slm import DBSLMEngine

engine = DBSLMEngine()
conversation_id = engine.start_conversation(user_id="demo")
print(engine.respond(conversation_id, "Remind me what we discussed."))
```

Use `train_from_text()` to ingest small corpora and extend the concept repository through the
`ConceptEngine` exposed on `engine.concepts`.

## Environment Setup

- Python 3.10+ is recommended. Create an isolated virtual environment if you plan to experiment:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt  # currently empty but preserves the workflow
  ```
- The CLI utilities default to storing everything under `var/db_slm.sqlite3`. Feel free to point them
  at any other SQLite path or even `:memory:` when using the programmatic API.

## Training CLI (`src/train.py`)

`train.py` is the canonical way to populate the Level 1 N-gram table from plain-text corpora. It wires
directly into `DBSLMEngine.train_from_text()` so Level 2/3 seed data is left intact while Level 1
statistics are extended.

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

## Inference CLI (`src/run.py`)

`run.py` spins up a conversational REPL backed by the database produced during training. It uses the
same `DBSLMEngine` façade as developers use from Python.

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

Useful flags:

- `--conversation`: Resume a previous conversation ID logged in `tbl_l2_conversations`. Omit to start a
  fresh session (the ID is printed on startup).
- `--user` / `--agent`: Override the identifiers written into Level 2 so you can distinguish sessions.
- `--max-turns`: Cap the number of turns before the REPL exits automatically. Handy for scripted demos.
- Commands inside the REPL: `:history` to print the current Level 2 context window, `:exit`/`:quit` (or
  `Ctrl+D`) to leave.

Because the CLI uses the exact same engine object, anything logged via `run.py` is immediately
available to downstream tooling (correction logging, concept payload providers, etc.).
