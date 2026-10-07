#!/usr/bin/env bash
#SBATCH -p blackwell
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=38
#SBATCH --mem=71000
#SBATCH --time=04:00:00
#SBATCH --job-name=ic-react-bw
#SBATCH --output=/playpen-shared/mohanz/Harness_Generalization/intercode/react-smoke/vllm-blackwell-%j.out
set -euo pipefail

envroot=/playpen-shared/mohanz/Harness_Generalization/Life-Harness/AgentBench/.venv-vllm-cu128
model=/playpen-shared/mohanz/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_DEEP_GEMM_E8M0=0
exec "$envroot/bin/vllm" serve "$model" \
  --host 0.0.0.0 --port 30019 --served-model-name qwen3-4b \
  --max-model-len 32768 --gpu-memory-utilization 0.80 \
  --enable-auto-tool-choice --tool-call-parser hermes
