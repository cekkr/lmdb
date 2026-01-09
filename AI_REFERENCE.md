# AI Reference

High-level deltas and operational context for lmdb. Keep this file aligned with README.md and
NEXT_STEPS.md so new agents inherit the latest state.

## Recent changes

- `src/train.py` defaults `--ngram-order` to auto (`0`), sampling the input corpus to pick a stable
  order; the resolved value is stored as `ngram_order` metadata for later reuse.
- Context-window sampling now supports auto windows/stride when the CLI knobs are left at `0`
  (train/infer windows and stride ratio).
- Decoder presence/frequency penalties auto-tune after evaluation probes/hold-outs unless explicit
  overrides are supplied; overrides lock the tuner.
- cheetah-db adds `PREDICT_INHERIT` so merged tokens can inherit prediction weights from their
  component tokens during training; run.py uses stored `ngram_order` when `--ngram-order 0`.

## Pointers

- Full CLI examples and flag details live in `README.md`.
- Cheetah-specific operational guidance is in `cheetah-db/AI_REFERENCE.md`.
