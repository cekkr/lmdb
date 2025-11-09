# lmdb
[Experimental] Database centric LLM

## Overview

This repository explores the database-native statistical language model (DB-SLM) designed in
`studies/CONCEPT.md`. Instead of tensors, all generation stages are persisted as relational tables:

- **Level 1 — Statistical N-grams (Aria-like):** Lookup tables for token probabilities.
- **Level 2 — Stateful Memory (MyRocks + InnoDB):** Conversation logs and correctional RAG.
- **Level 3 — Conceptual Generation (InnoDB):** Concept dictionaries, templates, and selection logic.

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
