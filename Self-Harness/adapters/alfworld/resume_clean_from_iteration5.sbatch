#!/usr/bin/env bash
#SBATCH -p all
#SBATCH --qos=scavenger
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --job-name=self-alf-clean-r5
#SBATCH --output=/playpen-shared/mohanz/Harness_Generalization/Self-Harness/runs/self-alf-clean-r5-%j.out
set -euo pipefail
SELF_ROOT=/home/mohanz/Harness_Generalization/Self-Harness
PYTHON=/home/mohanz/.venvs/agent/bin/python
RUN=/playpen-shared/mohanz/Harness_Generalization/Self-Harness/runs/self-alfworld-original-post4-v1
CURRENT="$SELF_ROOT/adapters/alfworld/alfworld_harness_surface.py"
export SELF_HARNESS_ALFWORLD_EVOLUTION_TASKS=140 SELF_HARNESS_ALFWORLD_HELDOUT_TASKS=134 SELF_HARNESS_ALFWORLD_SPLITS=evolution
export QWEN_API_BASE=http://unites4.ib:30017/v1 QWEN_MODEL=qwen3-8b QWEN_AGENT_NAME=qwen3-8b PROPOSER_API_BASE=http://unites4.ib:30017/v1 PROPOSER_MODEL=qwen3-8b
for I in $(seq 5 10); do
  export SELF_HARNESS_ALFWORLD_BASELINE="$CURRENT"
  W="$RUN/iteration-$I"
  test ! -e "$W" || { echo "refusing nonempty iteration work dir: $W" >&2; exit 2; }
  mkdir -p "$W"
  "$PYTHON" "$SELF_ROOT/workflow/scripts/run_self_harness_loop.py" --evaluator alfworld --eval-config "$SELF_ROOT/adapters/alfworld/alfworld_self.toml" --work-dir "$W" --surface "baseline=$CURRENT" --route-count 1 --diagnosis-command "$PYTHON $SELF_ROOT/adapters/alfworld/diagnose.py --work-dir {work_dir} --output {diagnosis}" --proposer-command "$PYTHON $SELF_ROOT/adapters/alfworld/propose.py --prompt {prompt} --response {response} --api-base http://unites4.ib:30017/v1 --model qwen3-8b --max-tokens 8192"
  CURRENT=$("$PYTHON" -c 'import json,sys; s=json.load(open(sys.argv[1])); b=next(x for x in s["branches"] if x["branch_id"]==s["active_branch_id"]); print(b["eval_surfaces"]["baseline"])' "$W/branch_state.json")
done
unset SELF_HARNESS_ALFWORLD_SPLITS
"$PYTHON" "$SELF_ROOT/adapters/alfworld/run_alfworld_eval.py" --candidate "$CURRENT" --output-dir "$RUN/final_heldout" --splits heldout --agent-max-tokens 2048
scancel 226379
