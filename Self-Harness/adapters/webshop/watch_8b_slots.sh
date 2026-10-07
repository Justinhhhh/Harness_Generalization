#!/usr/bin/env bash
set -euo pipefail
SELF=/home/mohanz/Harness_Generalization/Self-Harness
JOBS=(236587 236595 236596)
while true; do
  winner=""
  for job in "${JOBS[@]}"; do
    state=$(squeue -h -j "$job" -o '%T' 2>/dev/null || true)
    if [[ "$state" == RUNNING ]]; then
      winner=$job
      break
    fi
  done
  if [[ -n "$winner" ]]; then
    for job in "${JOBS[@]}"; do
      [[ "$job" == "$winner" ]] || scancel "$job" 2>/dev/null || true
    done
    srun --jobid="$winner" --overlap --ntasks=1 --cpus-per-task=1 bash -lc 'for i in $(seq 1 90); do curl -sf http://127.0.0.1:30017/v1/models >/dev/null && exit 0; sleep 5; done; exit 1'
    srun --jobid="$winner" --overlap --ntasks=1 --cpus-per-task=4 --job-name=webshop-current-active bash "$SELF/adapters/webshop/run_current_active_100.sh"
    scancel "$winner" 2>/dev/null || true
    exit 0
  fi
  sleep 20
done
