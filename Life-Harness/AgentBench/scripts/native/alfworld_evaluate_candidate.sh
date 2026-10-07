#!/usr/bin/env bash
set -euo pipefail

# Evaluate one Harness candidate through the normal AgentBench assigner.
# The worker must already be running with the same ALFWORLD_HARNESS_FILE.
# Usage: ALFWORLD_HARNESS_FILE=/path/candidate.py \
#   scripts/native/alfworld_evaluate_candidate.sh evolution <output-dir>

phase=${1:?Usage: $0 evolution|heldout output-dir}
out=${2:?Usage: $0 evolution|heldout output-dir}
limit=${3:-0}
case "$phase" in
  evolution) task=alfworld-evolution ;;
  smoke) task=alfworld-smoke10 ;;
  smoke25) task=alfworld-smoke25 ;;
  candidate) task=alfworld-candidate10 ;;
  candidate25) task=alfworld-candidate25 ;;
  candidate-evolution) task=alfworld-candidate-evolution ;;
  candidate-smoke) task=alfworld-candidate-smoke ;;
  heldout) task=alfworld-heldout ;;
  *) echo "phase must be smoke, smoke25, candidate-smoke, candidate, candidate25, candidate-evolution, evolution or heldout" >&2; exit 2 ;;
esac

if [[ "$phase" == evolution || "$phase" == smoke || "$phase" == smoke25 || "$phase" == candidate || "$phase" == candidate25 || "$phase" == candidate-evolution || "$phase" == candidate-smoke ]]; then
  split=alfworld_evolution
else
  split=alfworld_heldout
fi
range_block=""
if [[ "$limit" != 0 ]]; then
  range_block="        start: 0\n        end: $limit"
fi
task_concurrency=1
if [[ "$phase" == candidate-evolution ]]; then task_concurrency=${SELF_HARNESS_WORKERS:-2}; fi

test -n "${ALFWORLD_HARNESS_FILE:-}" || {
  echo 'ALFWORLD_HARNESS_FILE must point to a candidate Harness.py' >&2
  exit 2
}
test -f "$ALFWORLD_HARNESS_FILE" || {
  echo "candidate not found: $ALFWORLD_HARNESS_FILE" >&2
  exit 2
}

mkdir -p "$out"
config="$out/assign.yaml"
agent_config="$out/qwen.yaml"
controller_port=${AGENTRL_CONTROLLER_PORT:-15020}
qwen_base=${QWEN_API_BASE:-http://unites4.ib:30017/v1}
qwen_model=${QWEN_MODEL:-/playpen-shared/mohanz/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}
qwen_agent_name=${QWEN_AGENT_NAME:-qwen3-4b}
cat > "$agent_config" <<AGENT
${qwen_agent_name}:
  import: /home/mohanz/Harness_Generalization/Life-Harness/AgentBench/configs/agents/openai-chat.yaml
  parameters:
    name: ${qwen_agent_name}
    url: ${qwen_base}/chat/completions
    headers:
      Content-Type: application/json
      Authorization: Bearer EMPTY
    body:
      model: ${qwen_model}
      max_tokens: ${CHILD_MAX_TOKENS:-2048}
      temperature: 0.0
      chat_template_kwargs:
        enable_thinking: false
AGENT
cat > "$config" <<YAML
import: /home/mohanz/Harness_Generalization/Life-Harness/AgentBench/configs/assignments/definition.yaml
definition:
  task:
    overwrite:
      parameters:
        controller_address: http://127.0.0.1:$controller_port/api
        name: $task
        split: $split
        max_step: 50
$(printf '%b\n' "$range_block")
  agent:
    import:
      - $agent_config
concurrency:
  task:
    $task: $task_concurrency
  agent:
    ${qwen_agent_name}: $task_concurrency
assignments:
  - agent: [${qwen_agent_name}]
    task: [$task]
output: $out
trials: 1
YAML
PYTHONPATH=. .native/alfworld-venv/bin/python -m src.assigner -c "$config"
