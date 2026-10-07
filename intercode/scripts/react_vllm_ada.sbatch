#!/usr/bin/env bash
#SBATCH -p ada
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=60G
#SBATCH --time=04:00:00
#SBATCH --job-name=ic-react-vllm
#SBATCH --output=/playpen-shared/mohanz/Harness_Generalization/intercode/react-smoke/vllm-%j.out
set -euo pipefail

envroot=/playpen-shared/mohanz/Harness_Generalization/intercode/.venv-vllm-ada-cu124
model=/playpen-shared/mohanz/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
export VLLM_USE_FLASHINFER_SAMPLER=0
exec "$envroot/bin/vllm" serve "$model" \
  --host 0.0.0.0 --port 30018 --served-model-name qwen3-4b \
  --max-model-len 32768 --gpu-memory-utilization 0.80
