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

- `src/db_slm` now contains the initial Python scaffolding for the three-level DB-SLM blueprint described in `studies/CONCEPT.md`. `DBSLMEngine` wires Level 1 N-grams, Level 2 memory/corrections, and Level 3 concept prediction + verbalization.
- `requirements.txt` is currently empty (stdlib only) so downstream agents can add dependencies explicitly when needed.
- `src/train.py` ingests corpora into the SQLite store (default `var/db_slm.sqlite3`) with options for directory traversal, stdin ingestion, and database resets; `src/run.py` exposes a CLI REPL plus single-shot inference that records conversations inside Level 2.
