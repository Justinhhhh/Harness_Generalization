#!/usr/bin/env bash
# Run WebShop with the final ALFWorld Self-Harness baseline surface adapted
# through the existing WebShop session boundary.
set -euo pipefail

AB=/home/mohanz/Harness_Generalization/Life-Harness/AgentBench
SELF=/home/mohanz/Harness_Generalization/Self-Harness
OUT=/playpen-shared/mohanz/Harness_Generalization/Self-Harness/runs/webshop-alfworld-best-4b
export AGENTRL_CONTROLLER_PORT=19620 AGENTRL_WORKER_PORT=20620
export WEBSHOP_HARNESS_FILE="$SELF/adapters/webshop/current_active_harness_adapter.py"

mkdir -p "$OUT"
cd "$AB"
"$AB/.native/controller/agentrl" controller --host 127.0.0.1 --port 19620 --dashboard=false --long-timeout > "$OUT/controller.log" 2>&1 &
controller_pid=$!
bash scripts/native/worker.sh webshop webshop-test > "$OUT/worker.log" 2>&1 &
worker_pid=$!
cleanup() {
  kill "$worker_pid" "$controller_pid" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 90); do
  curl -sf http://127.0.0.1:20620/api/get_sessions >/dev/null && break
  sleep 2
done
curl -sf http://127.0.0.1:20620/api/get_sessions >/dev/null
PYTHONPATH=. .native/webshop-venv/bin/python -m src.assigner -c "$SELF/adapters/webshop/assign-alfworld-best-4b.yaml"
