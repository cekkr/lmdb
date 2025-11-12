#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME=${CHEETAH_SERVER_SESSION:-cheetahdb}

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  tmux kill-session -t "$SESSION_NAME"
fi

pkill -f cheetah-server 2>/dev/null || true
echo "Stopped cheetah-db server session (${SESSION_NAME}) and any stray cheetah-server processes."
