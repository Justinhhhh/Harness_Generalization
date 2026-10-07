#!/usr/bin/env bash
set -euo pipefail
SELF_ROOT=/home/mohanz/Harness_Generalization/Self-Harness
PYTHON=/home/mohanz/.venvs/agent/bin/python
ROOT=/playpen-shared/mohanz/Harness_Generalization/Self-Harness/runs/self-alfworld-original-post4-v1
CURRENT="$($PYTHON - "$ROOT/iteration-10/branch_state.json" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
b=next(x for x in s["branches"] if x["branch_id"]==s["active_branch_id"])
print(b["eval_surfaces"]["baseline"])
PY
)"
export SELF_HARNESS_ALFWORLD_EVOLUTION_TASKS=140 SELF_HARNESS_ALFWORLD_HELDOUT_TASKS=134 SELF_HARNESS_ALFWORLD_SPLITS=heldout
export QWEN_API_BASE=http://unites4.ib:30017/v1 QWEN_MODEL=qwen3-8b QWEN_AGENT_NAME=qwen3-8b
"$PYTHON" "$SELF_ROOT/adapters/alfworld/run_alfworld_eval.py" --candidate "$CURRENT" --output-dir "$ROOT/final_heldout" --splits heldout --agent-max-tokens 2048
scancel 226379
