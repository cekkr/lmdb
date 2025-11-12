#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
TS=${CHEETAH_SMOKE_TS:-$(date +%Y%m%d-%H%M%S)}
LOG=${CHEETAH_SMOKE_LOG:-var/eval_logs/cheetah_smoke_train_${TS}.log}
METRICS=${CHEETAH_SMOKE_METRICS:-var/eval_logs/cheetah_smoke_train_${TS}.json}
DB_PATH=${CHEETAH_SMOKE_DB:-/tmp/cheetah_smoke.sqlite3}
TIMEOUT=${CHEETAH_SMOKE_TIMEOUT:-1800}
rm -f "$DB_PATH"
CMD=(env DBSLM_BACKEND=${DBSLM_BACKEND:-cheetah-db} python3.11 src/train.py datasets/emotion_data.json \
  --db "$DB_PATH" \
  --ngram-order 3 \
  --eval-interval 2000 \
  --json-chunk-size 250 \
  --max-json-lines 1000 \
  --profile-ingest \
  --metrics-export "$METRICS")
if command -v timeout >/dev/null 2>&1; then
  CMD=(timeout "$TIMEOUT" "${CMD[@]}")
fi
printf '>> log=%s\n' "$LOG"
printf '>> metrics=%s\n' "$METRICS"
"${CMD[@]}" | tee "$LOG"
