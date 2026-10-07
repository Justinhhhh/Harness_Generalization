#!/usr/bin/env bash
#SBATCH -p all
#SBATCH --qos=scavenger
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=self-alfworld-10iter
#SBATCH --output=/playpen-shared/mohanz/Harness_Generalization/Self-Harness/runs/self-alfworld-10iter-%j.out
set -euo pipefail

SELF_ROOT=/home/mohanz/Harness_Generalization/Self-Harness
PYTHON=/home/mohanz/.venvs/agent/bin/python
RUN_ROOT=/playpen-shared/mohanz/Harness_Generalization/Self-Harness/runs/self-alfworld-original-self-harness-v1
export SELF_HARNESS_ALFWORLD_EVOLUTION_TASKS=140
export SELF_HARNESS_ALFWORLD_HELDOUT_TASKS=134
export SELF_HARNESS_ALFWORLD_BASELINE="$SELF_ROOT/adapters/alfworld/alfworld_harness_surface.py"
export QWEN_API_BASE=http://unites4.ib:30017/v1
export QWEN_MODEL=qwen3-8b
export QWEN_AGENT_NAME=qwen3-8b
export PROPOSER_API_BASE=http://unites4.ib:30017/v1
export PROPOSER_MODEL=qwen3-8b
export SELF_HARNESS_ALFWORLD_SPLITS=evolution

mkdir -p "$RUN_ROOT"
CURRENT_SURFACE="$SELF_ROOT/adapters/alfworld/alfworld_harness_surface.py"
# Iterations 1--2 are complete; iteration 3 reuses its completed baseline and
# train trajectories after the proposer-contract retry fix.
for iteration in $(seq 1 10); do
  ITER_ROOT="$RUN_ROOT/iteration-$iteration"
  mkdir -p "$ITER_ROOT"
  "$PYTHON" "$SELF_ROOT/workflow/scripts/run_self_harness_loop.py" \
    --evaluator alfworld \
    --eval-config "$SELF_ROOT/adapters/alfworld/alfworld_self.toml" \
    --work-dir "$ITER_ROOT" \
    --surface "baseline=$CURRENT_SURFACE" \
    --route-count 1 \
    --diagnosis-command "$PYTHON $SELF_ROOT/adapters/alfworld/diagnose.py --work-dir {work_dir} --output {diagnosis}" \
    --proposer-command "$PYTHON $SELF_ROOT/adapters/alfworld/propose.py --prompt {prompt} --response {response} --api-base http://unites4.ib:30017/v1 --model qwen3-8b --max-tokens 8192"
  CURRENT_SURFACE=$("$PYTHON" -c 'import json,sys; s=json.load(open(sys.argv[1])); b=next(x for x in s["branches"] if x["branch_id"] == s["active_branch_id"]); print(b["eval_surfaces"]["baseline"])' "$ITER_ROOT/branch_state.json")
  echo "completed self-harness iteration $iteration/10; active_surface=$CURRENT_SURFACE"
done

unset SELF_HARNESS_ALFWORLD_SPLITS
"$PYTHON" "$SELF_ROOT/adapters/alfworld/run_alfworld_eval.py" \
  --candidate "$CURRENT_SURFACE" \
  --output-dir "$RUN_ROOT/final_heldout-resume" \
  --splits heldout \
  --agent-max-tokens 2048
