#!/usr/bin/env bash
set -euo pipefail
SELF_ROOT=/home/mohanz/Harness_Generalization/Self-Harness
PYTHON=/home/mohanz/.venvs/agent/bin/python
W=/playpen-shared/mohanz/Harness_Generalization/Self-Harness/runs/self-alfworld-original-post4-v1/iteration-8
CURRENT="$SELF_ROOT/adapters/alfworld/alfworld_harness_surface.py"
export SELF_HARNESS_ALFWORLD_EVOLUTION_TASKS=140 SELF_HARNESS_ALFWORLD_HELDOUT_TASKS=134 SELF_HARNESS_ALFWORLD_SPLITS=evolution SELF_HARNESS_ALFWORLD_BASELINE="$CURRENT"
export QWEN_API_BASE=http://unites4.ib:30017/v1 QWEN_MODEL=qwen3-8b QWEN_AGENT_NAME=qwen3-8b PROPOSER_API_BASE=http://unites4.ib:30017/v1 PROPOSER_MODEL=qwen3-8b
exec "$PYTHON" "$SELF_ROOT/workflow/scripts/run_self_harness_loop.py" --evaluator alfworld --eval-config "$SELF_ROOT/adapters/alfworld/alfworld_self.toml" --work-dir "$W" --surface "baseline=$CURRENT" --route-count 1 --diagnosis-command "$PYTHON $SELF_ROOT/adapters/alfworld/diagnose.py --work-dir {work_dir} --output {diagnosis}" --proposer-command "$PYTHON $SELF_ROOT/adapters/alfworld/propose.py --prompt {prompt} --response {response} --api-base http://unites4.ib:30017/v1 --model qwen3-8b --max-tokens 8192"
