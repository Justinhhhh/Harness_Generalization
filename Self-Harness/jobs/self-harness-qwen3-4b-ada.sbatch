#!/usr/bin/env bash
#SBATCH -p ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=60000
#SBATCH --time=7-00:00:00
#SBATCH --job-name=self-harness-4b
#SBATCH --output=/home/mohanz/Harness_Generalization/Self-Harness/runs/self-harness-4b-%j.out
set -uo pipefail

MODEL=/playpen-shared/mohanz/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
VLLM=${SELF_HARNESS_VLLM:-/home/mohanz/Harness_Generalization/intercode/.venv-vllm-ada-cu124/bin/vllm}
LOG=/home/mohanz/Harness_Generalization/Self-Harness/runs/self-harness-4b-vllm-$SLURM_JOB_ID.log
export VLLM_USE_FLASHINFER_SAMPLER=0
export TMPDIR=/tmp/self-harness-4b-$SLURM_JOB_ID
export TEMP=$TMPDIR
export TMP=$TMPDIR
mkdir -p "$TMPDIR" /home/mohanz/Harness_Generalization/Self-Harness/runs

stopping=0
server_pid=
stop_service() {
  stopping=1
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  exit 0
}
trap stop_service TERM INT

while (( ! stopping )); do
  "$VLLM" serve "$MODEL" \
    --host 0.0.0.0 --port 30018 --served-model-name qwen3-4b \
    --max-model-len 32768 --gpu-memory-utilization 0.80 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    >> "$LOG" 2>&1 &
  server_pid=$!
  wait "$server_pid" || true
  server_pid=
  (( stopping )) || sleep 10
done
