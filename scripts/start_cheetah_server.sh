#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME=${CHEETAH_SERVER_SESSION:-cheetahdb}
LOG_PATH=${CHEETAH_SERVER_LOG:-$REPO_ROOT/var/cheetah-server.log}
BIN_PATH=${CHEETAH_SERVER_BIN:-$REPO_ROOT/cheetah-db/cheetah-server}
TIMEOUT_SECONDS=${CHEETAH_SERVER_TIMEOUT:-1800}
PID_PATH=${CHEETAH_SERVER_PID_FILE:-$REPO_ROOT/var/cheetah-server-${SESSION_NAME}.pid}

if [[ ! -x "$BIN_PATH" ]]; then
  echo "cheetah-server binary not found at $BIN_PATH" >&2
  exit 1
fi

if ! [[ "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "CHEETAH_SERVER_TIMEOUT must be a positive integer (seconds)." >&2
  exit 1
fi

mkdir -p "$(dirname -- "$LOG_PATH")"
mkdir -p "$(dirname -- "$PID_PATH")"
if [[ -f "$PID_PATH" ]]; then
  old_pid=$(<"$PID_PATH")
  if [[ "$old_pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "cheetah-server PID $old_pid from $PID_PATH is still running." >&2
    exit 1
  fi
  rm -f "$PID_PATH"
fi
if command -v timeout >/dev/null 2>&1; then
  timeout_cmd="timeout $TIMEOUT_SECONDS"
elif command -v gtimeout >/dev/null 2>&1; then
  timeout_cmd="gtimeout $TIMEOUT_SECONDS"
else
  timeout_cmd="perl -e 'alarm shift; exec @ARGV' $TIMEOUT_SECONDS"
fi
cmd="cd \"$REPO_ROOT/cheetah-db\" && echo \$\$ > \"$PID_PATH\" && exec $timeout_cmd env CHEETAH_HEADLESS=1 \"$BIN_PATH\" >> \"$LOG_PATH\" 2>&1"

if command -v screen >/dev/null 2>&1; then
  if (screen -ls 2>/dev/null || true) | grep -Eq "[.]${SESSION_NAME}[[:space:]]"; then
    echo "screen session $SESSION_NAME already running. Use scripts/stop_cheetah_server.sh first." >&2
    exit 1
  fi
  if screen -dmS "$SESSION_NAME" /bin/bash -lc "$cmd"; then
    printf 'cheetah-db server started (backend=screen, session=%s, timeout=%ss, log=%s, pid_file=%s)\n' \
      "$SESSION_NAME" "$TIMEOUT_SECONDS" "$LOG_PATH" "$PID_PATH"
    exit 0
  fi
  echo "screen could not start the session; trying tmux fallback." >&2
fi

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session $SESSION_NAME already running. Use scripts/stop_cheetah_server.sh first." >&2
    exit 1
  fi
  tmux new-session -d -s "$SESSION_NAME" "$cmd"
  printf 'cheetah-db server started (backend=tmux fallback, session=%s, timeout=%ss, log=%s, pid_file=%s)\n' \
    "$SESSION_NAME" "$TIMEOUT_SECONDS" "$LOG_PATH" "$PID_PATH"
  exit 0
fi

echo "Neither screen nor tmux is available; cheetah-db was not started." >&2
exit 1
