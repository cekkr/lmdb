# AI Reference

High-level deltas and operational context for lmdb. Keep this file aligned with README.md and
NEXT_STEPS.md so new agents inherit the latest state.

## Recent changes

- Sentence punctuation splitting during training is now disabled by default; enable it with
  `--sentence-splitting` or `DBSLM_SENTENCE_SPLIT=1` when needed.
- `src/train.py` defaults `--ngram-order` to auto (`0`), sampling the input corpus to pick a stable
  order; the resolved value is stored as `ngram_order` metadata for later reuse.
- Context-window sampling now supports auto windows/stride when the CLI knobs are left at `0`
  (train/infer windows and stride ratio).
- Decoder presence/frequency penalties auto-tune after evaluation probes/hold-outs unless explicit
  overrides are supplied; overrides lock the tuner.
- cheetah-db now supports batched/async `PREDICT_INHERIT` jobs (plus `STATUS`/`FETCH`), and the
  trainer queues merged-token inheritance through those batches when available.
- Pair trie terminals can be hidden via `PAIR_SET_HIDDEN`; `PAIR_SCAN`/`PAIR_REDUCE`/`PAIR_SUMMARY`
  accept `include_hidden=1` to surface cached joins without polluting default namespace scans.
- Decoder scoring now flows through `TokenScoringPipeline` (`src/db_slm/scoring.py`) with optional
  `ScoreObserver` snapshots wired through `DBSLMEngine.respond()` and `issue_prompt()` for
  statistical debugging.
- cheetah-db now resolves `PAIR_REDUCE` modes through a reducer registry (`cheetah-db/reducers.go`)
  so new reducer implementations can be added without editing command dispatch.

## Pointers

- Full CLI examples and flag details live in `README.md`.
- Cheetah-specific operational guidance is in `cheetah-db/AI_REFERENCE.md`.
