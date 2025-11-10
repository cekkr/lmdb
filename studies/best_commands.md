# Best Command Presets

Documented command lines for repeatable, efficient DB-SLM training workflows. Adjust dataset paths or database targets as needed, but keep the relative flag ratios so throughput stays high.

## Efficient Smoke-Train

Use when validating ingestion + evaluation together without wasting hours on probes.

```bash
python3.11 src/train.py datasets/emotion_data.json \
  --ngram-order 5 \
  --recursive \
  --reset \
  --json-chunk-size 500 \
  --eval-interval 5000 \
  --eval-samples 1 \
  --eval-pool-size 200 \
  --chunk-eval-percent 1.0 \
  --profile-ingest \
  --metrics-export var/eval_logs/train-$(date +%Y%m%d-%H%M%S).json
```

- Larger chunks amortize the MKNS smoother rebuilds; aim for ≥400 unless memory says otherwise.
- `--eval-interval 5000` keeps evaluation overhead below ~2% of wall clock while still providing checkpoints every ~250k tokens for the 500-token chunks.
- The `--profile-ingest` flag records chunk duration and RSS deltas; append the best run to `studies/datasets.md`.
- When sweeping repeat penalties, append `--decoder-presence-penalty <value> --decoder-frequency-penalty <value>` so the overrides propagate to the held-out probes and land in the metrics metadata for later comparisons.

## Queue-Drain Retrain

Use to clear `var/eval_logs/quality_retrain_queue.jsonl` when it approaches 150 flagged samples.

```bash
python3.11 src/train.py var/eval_logs/quality_retrain_queue.jsonl \
  --db var/db_slm.sqlite3 \
  --ngram-order 5 \
  --json-chunk-size 200 \
  --eval-interval 0 \
  --chunk-eval-percent 0 \
  --profile-ingest \
  --metrics-export var/eval_logs/train-queue-drain-$(date +%Y%m%d-%H%M%S).json
```

- Disable streaming probes so all wall clock goes to ingesting the remediation samples.
- Keep `--json-chunk-size` around 200 because the queue entries are short; larger batches make it easier to correlate improvements once reintroduced in the main corpus.

## Baseline Throughput Probe

Use after system or dataset changes to quickly verify ingestion speed without touching the primary database.

```bash
python3.11 src/train.py datasets/emotion_data.json \
  --db var/smoke-train.sqlite3 \
  --ngram-order 5 \
  --json-chunk-size 600 \
  --eval-interval 0 \
  --chunk-eval-percent 0 \
  --profile-ingest \
  --max-json-lines 3000 \
  --metrics-export var/eval_logs/train-baseline-$(date +%Y%m%d-%H%M%S).json
```

- `--max-json-lines` caps runtime (~5 chunks at 600 lines each) while still producing reliable RSS/latency data.
- Run this immediately after dependency upgrades or schema tweaks; compare the tokens/s figures with the previous baseline stored in `AI_REFERENCE.md`.
