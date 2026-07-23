#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME=${CHEETAH_SMOKE_SESSION:-cheetah_smoke}
cd "$REPO_ROOT"

ts=$(date +%Y%m%d-%H%M%S)
log_path=${CHEETAH_SMOKE_LOG:-$REPO_ROOT/var/eval_logs/cheetah_smoke_train_${ts}.log}
metrics_path=${CHEETAH_SMOKE_METRICS:-$REPO_ROOT/var/eval_logs/cheetah_smoke_train_${ts}.json}
db_path=${CHEETAH_SMOKE_DB:-/tmp/cheetah_smoke_${ts}.sqlite3}
timeout_s=${CHEETAH_SMOKE_TIMEOUT:-1800}

cmd="cd \"$REPO_ROOT\" && CHEETAH_SMOKE_LOG=\"$log_path\" CHEETAH_SMOKE_METRICS=\"$metrics_path\" CHEETAH_SMOKE_DB=\"$db_path\" CHEETAH_SMOKE_TIMEOUT=$timeout_s /bin/bash scripts/run_cheetah_smoke.sh"
if command -v screen >/dev/null 2>&1; then
  if (screen -ls 2>/dev/null || true) | grep -Eq "[.]${SESSION_NAME}[[:space:]]"; then
    echo "screen session $SESSION_NAME already running; refusing to replace it." >&2
    exit 1
  fi
  if screen -dmS "$SESSION_NAME" /bin/bash -lc "$cmd"; then
    printf 'backend=screen log=%s\n' "$log_path"
    exit 0
  fi
  echo "screen could not start the smoke session; trying tmux fallback." >&2
fi

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session $SESSION_NAME already running; refusing to replace it." >&2
    exit 1
  fi
  tmux new-session -d -s "$SESSION_NAME" "$cmd"
  printf 'backend=tmux-fallback log=%s\n' "$log_path"
  exit 0
fi

echo "Neither screen nor tmux is available; smoke run was not started." >&2
exit 1
