#!/usr/bin/env bash
set -euo pipefail
ada_job="$1"
bw_job="$2"
log_file="/playpen-shared/mohanz/Harness_Generalization/intercode/react-smoke/qwen4b-pair-watch.log"
while true; do
  ada_state=$(squeue -h -j "$ada_job" -o '%T' 2>/dev/null || true)
  bw_state=$(squeue -h -j "$bw_job" -o '%T' 2>/dev/null || true)
  if [[ "$ada_state" == RUNNING ]]; then
    [[ "$bw_state" == PENDING ]] && scancel "$bw_job" >>"$log_file" 2>&1 || true
    echo "$(date -u +%FT%TZ) ada=$ada_job running; cancelled blackwell=$bw_job" >>"$log_file"
    exit 0
  fi
  if [[ "$bw_state" == RUNNING ]]; then
    [[ "$ada_state" == PENDING ]] && scancel "$ada_job" >>"$log_file" 2>&1 || true
    echo "$(date -u +%FT%TZ) blackwell=$bw_job running; cancelled ada=$ada_job" >>"$log_file"
    exit 0
  fi
  if [[ -z "$ada_state" || -z "$bw_state" ]]; then
    echo "$(date -u +%FT%TZ) pair ended: ada=$ada_state blackwell=$bw_state" >>"$log_file"
    exit 0
  fi
  sleep 15
done
