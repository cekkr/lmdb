#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
TS=${CHEETAH_SMOKE_TS:-$(date +%Y%m%d-%H%M%S)}
LOG=${CHEETAH_SMOKE_LOG:-var/eval_logs/cheetah_smoke_train_${TS}.log}
METRICS=${CHEETAH_SMOKE_METRICS:-var/eval_logs/cheetah_smoke_train_${TS}.json}
DB_PATH=${CHEETAH_SMOKE_DB:-/tmp/cheetah_smoke.sqlite3}
TIMEOUT=${CHEETAH_SMOKE_TIMEOUT:-1800}
MAX_JSON_LINES=${CHEETAH_SMOKE_MAX_JSON_LINES:-1000}
JSON_CHUNK_SIZE=${CHEETAH_SMOKE_JSON_CHUNK_SIZE:-250}
EVAL_INTERVAL=${CHEETAH_SMOKE_EVAL_INTERVAL:-2000}
DATABASE=${CHEETAH_SMOKE_DATABASE:-emotion_smoke_${TS}}
rm -f "$DB_PATH"
CMD=(env DBSLM_BACKEND=${DBSLM_BACKEND:-cheetah-db} DBSLM_CHEETAH_DATABASE="$DATABASE" python3.11 src/train.py datasets/emotion_data.json \
  --db "$DB_PATH" \
  --ngram-order 3 \
  --eval-interval "$EVAL_INTERVAL" \
  --json-chunk-size "$JSON_CHUNK_SIZE" \
  --max-json-lines "$MAX_JSON_LINES" \
  --profile-ingest \
  --metrics-export "$METRICS")
if command -v timeout >/dev/null 2>&1; then
  CMD=(timeout "$TIMEOUT" "${CMD[@]}")
elif command -v gtimeout >/dev/null 2>&1; then
  CMD=(gtimeout "$TIMEOUT" "${CMD[@]}")
else
  CMD=(perl -e 'alarm shift; exec @ARGV' "$TIMEOUT" "${CMD[@]}")
fi
printf '>> log=%s\n' "$LOG"
printf '>> metrics=%s\n' "$METRICS"
printf '>> cheetah_database=%s\n' "$DATABASE"
"${CMD[@]}" | tee "$LOG"
