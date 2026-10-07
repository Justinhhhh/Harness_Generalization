#!/usr/bin/env bash
# Qwen3.5-9B service for Meta-Harness WebShop evaluation.
# This uses the Blackwell/CUDA 12.8 vLLM environment; the Ada vLLM 0.8.5
# environment does not provide the required model support.
#SBATCH --partition=blackwell
#SBATCH --qos=normal
#SBATCH --nodelist=unites8.ib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=58G
#SBATCH --time=04:00:00
#SBATCH --job-name=qwen3.5-9b-alfworld
#SBATCH --output=/home/mohanz/Harness_Generalization/Self-Harness/runs/qwen3.5-9b-alfworld-%j.out
set -euo pipefail

export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_DEEP_GEMM_E8M0=0

exec /home/mohanz/Harness_Generalization/Life-Harness/AgentBench/.venv-vllm-cu128/bin/vllm serve \
  /home/mohanz/.cache/huggingface/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --host 0.0.0.0 --port 31615 \
  --served-model-name qwen3.5-9b \
  --max-model-len 32768 --gpu-memory-utilization 0.80 \
  --enable-auto-tool-choice --tool-call-parser hermes
