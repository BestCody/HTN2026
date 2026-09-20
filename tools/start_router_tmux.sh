#!/usr/bin/env bash
set -euo pipefail

/usr/bin/tmux kill-session -t moira-router 2>/dev/null || true
/usr/bin/tmux new-session -d -s moira-router
/usr/bin/tmux send-keys -t moira-router \
  'cd /c/Users/louha/OneDrive/Desktop/HTN2026 && bash tools/run_router_training_tmux.sh' \
  C-m
/usr/bin/tmux list-sessions
