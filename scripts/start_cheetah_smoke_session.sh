#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME=${CHEETAH_SMOKE_SESSION:-cheetah_smoke}
cd "$REPO_ROOT"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  tmux kill-session -t "$SESSION_NAME"
fi

ts=$(date +%Y%m%d-%H%M%S)
log_path="$REPO_ROOT/var/eval_logs/cheetah_smoke_train_${ts}.log"
metrics_path="$REPO_ROOT/var/eval_logs/cheetah_smoke_train_${ts}.json"
db_path="/tmp/cheetah_smoke_${ts}.sqlite3"
timeout_s=${CHEETAH_SMOKE_TIMEOUT:-1800}

cmd="cd $REPO_ROOT && CHEETAH_SMOKE_LOG=\"$log_path\" CHEETAH_SMOKE_METRICS=\"$metrics_path\" CHEETAH_SMOKE_DB=\"$db_path\" CHEETAH_SMOKE_TIMEOUT=$timeout_s /bin/bash scripts/run_cheetah_smoke.sh"
tmux new-session -d -s "$SESSION_NAME" "$cmd"
printf '%s\n' "$log_path"
