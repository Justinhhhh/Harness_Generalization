#!/usr/bin/env bash
# This job is submitted with afterok against the ALFWorld evolution job.
# It runs only after all evolution iterations and the final heldout evaluator
# have exited successfully.
#SBATCH -p all
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:05:00
#SBATCH --job-name=release-alf-vllm
#SBATCH --output=/playpen-shared/mohanz/Harness_Generalization/Self-Harness/runs/release-alf-vllm-%j.out
set -euo pipefail

scancel 226379
