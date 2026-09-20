#!/usr/bin/env bash
set -u

cd /c/Users/louha/OneDrive/Desktop/HTN2026 || exit 90
mkdir -p outputs/router_training
script_path="$(cygpath -w tools/run_router_training.ps1)"
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "$script_path" \
  > outputs/router_training/tmux.log 2>&1
status=$?
printf '%s\n' "$status" > outputs/router_training/exit_code.txt
printf 'Router pipeline exited with code %s\n' "$status"
sleep 300
exit "$status"
