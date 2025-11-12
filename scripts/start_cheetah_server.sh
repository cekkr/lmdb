#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME=${CHEETAH_SERVER_SESSION:-cheetahdb}
LOG_PATH=${CHEETAH_SERVER_LOG:-$REPO_ROOT/var/cheetah-server-linux.log}
BIN_PATH=${CHEETAH_SERVER_BIN:-$REPO_ROOT/cheetah-db/cheetah-server-linux}

if [[ ! -x "$BIN_PATH" ]]; then
  echo "cheetah-server binary not found at $BIN_PATH" >&2
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session $SESSION_NAME already running. Use scripts/stop_cheetah_server.sh first." >&2
  exit 1
fi

mkdir -p "$(dirname -- "$LOG_PATH")"
cmd="cd $REPO_ROOT/cheetah-db && CHEETAH_HEADLESS=1 \"$BIN_PATH\" >> \"$LOG_PATH\" 2>&1"
tmux new-session -d -s "$SESSION_NAME" "$cmd"
printf 'cheetah-db server started (session=%s, log=%s)\n' "$SESSION_NAME" "$LOG_PATH"
