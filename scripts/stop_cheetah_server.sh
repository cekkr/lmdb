#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME=${CHEETAH_SERVER_SESSION:-cheetahdb}
PID_PATH=${CHEETAH_SERVER_PID_FILE:-$REPO_ROOT/var/cheetah-server-${SESSION_NAME}.pid}
stopped=0

if [[ -f "$PID_PATH" ]]; then
  server_pid=$(<"$PID_PATH")
  if [[ "$server_pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid"
    for _ in 1 2 3 4 5; do
      if ! kill -0 "$server_pid" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -KILL "$server_pid"
    fi
    stopped=1
  fi
  rm -f "$PID_PATH"
fi

if command -v screen >/dev/null 2>&1 && (screen -ls 2>/dev/null || true) | grep -Eq "[.]${SESSION_NAME}[[:space:]]"; then
  screen -S "$SESSION_NAME" -X quit
  stopped=1
fi
if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  tmux kill-session -t "$SESSION_NAME"
  stopped=1
fi

if [[ "$stopped" -eq 1 ]]; then
  echo "Stopped cheetah-db server session (${SESSION_NAME})."
else
  echo "No cheetah-db session named ${SESSION_NAME} was running."
fi
