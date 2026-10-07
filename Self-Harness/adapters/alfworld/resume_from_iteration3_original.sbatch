#!/usr/bin/env bash
#SBATCH -p all
#SBATCH --qos=scavenger
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=self-alf-resume3
#SBATCH --output=/playpen-shared/mohanz/Harness_Generalization/Self-Harness/runs/self-alf-resume3-%j.out
set -euo pipefail
SELF_ROOT=/home/mohanz/Harness_Generalization/Self-Harness
PYTHON=/home/mohanz/.venvs/agent/bin/python
OLD=/playpen-shared/mohanz/Harness_Generalization/Self-Harness/runs/self-alfworld-10iter-evol-only-v2/iteration-3
RUN=/playpen-shared/mohanz/Harness_Generalization/Self-Harness/runs/self-alfworld-original-resume3-v2
SURFACE="$SELF_ROOT/adapters/alfworld/alfworld_harness_surface.py"
export SELF_HARNESS_ALFWORLD_EVOLUTION_TASKS=140 SELF_HARNESS_ALFWORLD_HELDOUT_TASKS=134 SELF_HARNESS_ALFWORLD_SPLITS=evolution
export QWEN_API_BASE=http://unites4.ib:30017/v1 QWEN_MODEL=qwen3-8b QWEN_AGENT_NAME=qwen3-8b PROPOSER_API_BASE=http://unites4.ib:30017/v1 PROPOSER_MODEL=qwen3-8b
mkdir -p "$RUN/iteration-3/baseline_eval"
cp -a "$OLD/baseline_eval/result.json" "$RUN/iteration-3/baseline_eval/result.json"
"$PYTHON" "$SELF_ROOT/workflow/scripts/run_self_harness_loop.py" --evaluator alfworld --eval-config "$SELF_ROOT/adapters/alfworld/alfworld_self.toml" --work-dir "$RUN/iteration-3" --surface "baseline=$SURFACE" --route-count 1 --diagnosis "$OLD/branches/baseline/diagnosis/diagnosis.md" --proposer-command "$PYTHON $SELF_ROOT/adapters/alfworld/propose.py --prompt {prompt} --response {response} --api-base http://unites4.ib:30017/v1 --model qwen3-8b --max-tokens 8192"
CURRENT=$("$PYTHON" -c 'import json,sys; s=json.load(open(sys.argv[1])); b=next(x for x in s["branches"] if x["branch_id"]==s["active_branch_id"]); print(b["eval_surfaces"]["baseline"])' "$RUN/iteration-3/branch_state.json")
for I in $(seq 4 10); do
  W="$RUN/iteration-$I"; mkdir -p "$W"
  "$PYTHON" "$SELF_ROOT/workflow/scripts/run_self_harness_loop.py" --evaluator alfworld --eval-config "$SELF_ROOT/adapters/alfworld/alfworld_self.toml" --work-dir "$W" --surface "baseline=$CURRENT" --route-count 1 --diagnosis-command "$PYTHON $SELF_ROOT/adapters/alfworld/diagnose.py --work-dir {work_dir} --output {diagnosis}" --proposer-command "$PYTHON $SELF_ROOT/adapters/alfworld/propose.py --prompt {prompt} --response {response} --api-base http://unites4.ib:30017/v1 --model qwen3-8b --max-tokens 8192"
  CURRENT=$("$PYTHON" -c 'import json,sys; s=json.load(open(sys.argv[1])); b=next(x for x in s["branches"] if x["branch_id"]==s["active_branch_id"]); print(b["eval_surfaces"]["baseline"])' "$W/branch_state.json")
done
unset SELF_HARNESS_ALFWORLD_SPLITS
"$PYTHON" "$SELF_ROOT/adapters/alfworld/run_alfworld_eval.py" --candidate "$CURRENT" --output-dir "$RUN/final_heldout" --splits heldout --agent-max-tokens 2048
