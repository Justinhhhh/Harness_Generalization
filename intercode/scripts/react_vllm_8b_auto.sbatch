#!/usr/bin/env bash
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=60G
#SBATCH --time=04:00:00
#SBATCH --job-name=ic-react-8b
#SBATCH --output=/playpen-shared/mohanz/Harness_Generalization/intercode/react-smoke/vllm-8b-%j.out
set -euo pipefail

gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
model=/playpen-shared/mohanz/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5
export VLLM_USE_FLASHINFER_SAMPLER=0
if [[ "$gpu" == *"RTX A6000"* ]]; then
    envroot=/playpen-shared/mohanz/Harness_Generalization/intercode/.venv-vllm-ada-cu124
else
    envroot=/playpen-shared/mohanz/Harness_Generalization/Life-Harness/AgentBench/.venv-vllm-cu128
    export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM_E8M0=0
fi
exec "$envroot/bin/vllm" serve "$model" --host 0.0.0.0 --port 30021 \
  --served-model-name qwen3-8b --max-model-len 32768 --gpu-memory-utilization 0.80
