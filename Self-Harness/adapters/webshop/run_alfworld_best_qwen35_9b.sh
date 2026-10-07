#!/usr/bin/env bash
set -euo pipefail

AB=/home/mohanz/Harness_Generalization/Life-Harness/AgentBench
SELF=/home/mohanz/Harness_Generalization/Self-Harness
OUT=/playpen-shared/mohanz/Harness_Generalization/Self-Harness/runs/webshop-alfworld-best-qwen35-9b
export AGENTRL_CONTROLLER_PORT=19630 AGENTRL_WORKER_PORT=20630
export WEBSHOP_HARNESS_FILE="$SELF/adapters/webshop/current_active_harness_adapter.py"

mkdir -p "$OUT"
cd "$AB"
"$AB/.native/controller/agentrl" controller --host 127.0.0.1 --port 19630 --dashboard=false --long-timeout > "$OUT/controller.log" 2>&1 &
controller_pid=$!
bash scripts/native/worker.sh webshop webshop-test > "$OUT/worker.log" 2>&1 &
worker_pid=$!
cleanup() { kill "$worker_pid" "$controller_pid" 2>/dev/null || true; }
trap cleanup EXIT

for _ in $(seq 1 90); do
  curl -sf http://127.0.0.1:20630/api/get_sessions >/dev/null && break
  sleep 2
done
curl -sf http://127.0.0.1:20630/api/get_sessions >/dev/null
PYTHONPATH=. .native/webshop-venv/bin/python -m src.assigner -c "$SELF/adapters/webshop/assign-alfworld-best-qwen35-9b.yaml"
