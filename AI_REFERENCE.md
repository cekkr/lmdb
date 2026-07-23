# AI Reference

High-level deltas and operational context for lmdb. Keep this file aligned with README.md and
NEXT_STEPS.md so new agents inherit the latest state.

## Recent changes

- The Cheetah submodule is pinned to the local upstream repair commit `6866be9`. `ValuesTable`
  reserves append slots with an atomic high-water mark, and pair mutations share one database lock,
  preventing equal-size inserts and concurrent shared-prefix `PAIR_SET` calls from overwriting data.
- `CheetahHotPathAdapter` uses the current `JOB submit command=<base64>` → `JOB status id=...` →
  `JOB fetch id=...` reducer flow, falling back to the legacy `PAIR_REDUCE_ASYNC` aliases only when
  an older server rejects `JOB`. Reducer payloads must also have Cheetah's storage base64 layer
  removed before DB-SLM deserialization.
- Cheetah server/smoke helpers launch bounded `screen` sessions, fall back to `tmux` only when
  necessary, and track the exact server PID. The smoke runner accepts named database, row/chunk,
  evaluation interval, timeout, log, and metrics overrides.
- Prompt-tag exhaustion no longer returns the final invalid decoder candidate: the engine clears
  rejected token IDs and emits a scaffold-free response backstop. Tagged formatting also removes an
  already-framed terminal `|RESPONSE|:` before wrapping, avoiding nested `|USER|:` frames.
- The 2026-07-23 bounded Cheetah-only emotion validation ingested 20 records / 380,200 tokens /
  380,198 n-grams in 206.82 seconds, completed prediction updates with the adapter still active, and
  reused that named Cheetah database for inference. See `studies/BENCHMARKS.md`.
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
- cheetah-db now resolves `PAIR_REDUCE` modes through a reducer registry (`cheetah-db/src/reducers.go`)
  so new reducer implementations can be added without editing command dispatch.

## Pointers

- Full CLI examples and flag details live in `README.md`.
- Cheetah is maintained as the `cheetah-db/` submodule (`https://github.com/cekkr/cheetah`); read `cheetah-db/AGENTS.md` and keep generic database-server fixes committed upstream, updating this gitlink afterward.
- `NEXT_STEPS.md` is the active backlog; the old reducer-disable/smoke-hang items are resolved and
  replaced by an evaluation-enabled 250/1,000-record scale-up run.
