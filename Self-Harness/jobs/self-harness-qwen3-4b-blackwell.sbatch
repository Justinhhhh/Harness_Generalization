#!/usr/bin/env bash
#SBATCH -p blackwell
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=60000
#SBATCH --time=7-00:00:00
#SBATCH --job-name=self-harness-4b-blackwell
#SBATCH --output=/home/mohanz/Harness_Generalization/Self-Harness/runs/self-harness-4b-%j.out
set -uo pipefail
export SELF_HARNESS_VLLM=/home/mohanz/Harness_Generalization/Life-Harness/AgentBench/.venv-vllm-cu128/bin/vllm
exec /home/mohanz/Harness_Generalization/Self-Harness/jobs/self-harness-qwen3-4b-ada.sbatch
