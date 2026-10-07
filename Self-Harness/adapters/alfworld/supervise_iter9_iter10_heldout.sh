#!/usr/bin/env bash
set -euo pipefail
SELF=/playpen-shared/mohanz/Harness_Generalization/Self-Harness
LOG="$SELF/runs/self-alf-supervisor.log"
while squeue -h -j 226379 -s -o '%j' | rg -q '^self-alf-r9-original$'; do sleep 30; done
srun --jobid=226379 --overlap --ntasks=1 --cpus-per-task=4 --job-name=self-alf-r10-original bash "$SELF/adapters/alfworld/resume_original_iteration10.sh" >>"$LOG" 2>&1
srun --jobid=226379 --overlap --ntasks=1 --cpus-per-task=4 --job-name=self-alf-heldout bash "$SELF/adapters/alfworld/run_final_heldout_after_iter10.sh" >>"$LOG" 2>&1
