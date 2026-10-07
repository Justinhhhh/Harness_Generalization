# Resume Qwen3-4B SWE-bench Lite Meta-Harness

## Run state

Run directory:

```text
/home/mohanz/Harness_Generalization/meta-harness/swebench_lite/runs/qwen3-4b-swe-lite-metaharness-v2-20261005
```

Do not change the `--run` value. The run is resumable from any completed or
partially completed iteration. The worker's `--resume` preserves existing
trajectory records; the orchestrator reconstructs the best candidate from the
completed `evaluation.json` files.

## 1. Request a debug GPU

Submit the existing 4B debug vLLM job:

```bash
sbatch --parsable \
  /home/mohanz/Harness_Generalization/meta-harness/swebench_lite/jobs/serve_qwen3_4b_debug.sbatch
```

The job uses one GPU for four hours and serves Qwen3-4B on port `31595`. It
automatically selects the Blackwell or Ada vLLM environment based on the GPU.
Wait until Slurm shows `RUNNING` and record the allocated node:

```bash
squeue -j JOBID -o "%.18i %.12T %.10M %.12l %.24j %.16R"
export VLLM_NODE=NODE_FROM_SQUEUE
curl "http://${VLLM_NODE}:31595/v1/models"
```

The response must contain the served model `qwen3-4b` before starting the
orchestrator.

## 2. Resume the 4B run

Run from the repository root:

```bash
cd /home/mohanz/Harness_Generalization
export OPENAI_API_KEY=EMPTY
export QWEN_API_BASE="http://${VLLM_NODE}:31595/v1"
export OPENAI_API_BASE="$QWEN_API_BASE"

/home/mohanz/Harness_Generalization/SWE-bench/.venv/bin/python \
  /home/mohanz/Harness_Generalization/meta-harness/swebench_lite/formal_orchestrator.py \
  --run qwen3-4b-swe-lite-metaharness-v2-20261005 \
  --iterations 10 \
  --evolution-tasks 100 \
  --heldout-tasks 100 \
  --workers 2 \
  --evolution-workers 4 \
  --api-base "$QWEN_API_BASE"
```

Here `--workers 2` is used for smoke and heldout, while
`--evolution-workers 4` is the evolution-task concurrency. It is not the vLLM
model batch size and it does not create four proposers.

## 3. Verify resume

```bash
ps -eo pid,etime,cmd | rg 'formal_orchestrator|run_local_split'
python - <<'PY'
import json
from pathlib import Path

root = Path('/home/mohanz/Harness_Generalization/meta-harness/swebench_lite/runs/qwen3-4b-swe-lite-metaharness-v2-20261005')
for iteration in sorted(root.glob('iteration-*')):
    path = iteration / 'trajectories' / 'preds.json'
    if path.exists():
        print(iteration.name, len(json.loads(path.read_text())))
PY
```

The next incomplete iteration should increase monotonically. Do not delete
`preds.json`, `evaluation.json`, candidate manifests, or trajectory files.

## Slurm timeout recovery

Debug jobs are limited to four hours. If the vLLM job times out or is
cancelled, the orchestrator may remain as an orphan with no usable API. Stop
only the stale 4B processes, submit a new debug job, verify `/v1/models`, and
run the same command above. Resume will continue from the last written task.

```bash
ps -eo pid,cmd | rg 'qwen3-4b-swe-lite-metaharness-v2-20261005'
kill STALE_4B_ORCHESTRATOR_PID STALE_4B_RUNNER_PID
```

Do not kill independent 8B runs or unrelated Slurm jobs. If the debug job is
pending with `QOSGrpGRES`, leave it pending until a GPU is assigned rather than
starting a second copy of the same 4B run.
