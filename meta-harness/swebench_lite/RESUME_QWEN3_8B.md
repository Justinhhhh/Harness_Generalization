# Resume Qwen3-8B Meta-Harness

## Current stopped state

The Qwen3-8B orchestrator and worker are currently stopped. Existing results
are preserved:

```text
iteration 1: 0.01
iteration 2: 0.02
iteration 3: 0.00
iteration 4: 0.00
iteration 5: partial; resumable
```

There are currently no Qwen3-8B Meta-Harness processes running. Do not delete
the run directory before resuming.

Run directory:

```text
/home/mohanz/Harness_Generalization/meta-harness/swebench_lite/runs/qwen3-8b-swe-lite-metaharness-20261006
```

First check which node is serving vLLM:

```bash
squeue -u "$USER" -o "%.10i %.12P %.24j %.10T %.18R"
```

The server must expose `/v1/models` on port `31606`. Replace `VLLM_NODE` below with
the allocated node, for example `unites8.ib`:

```bash
export VLLM_NODE=unites8.ib
curl "http://${VLLM_NODE}:31606/v1/models"
```

Then resume the existing run from the repository root:

```bash
cd /home/mohanz/Harness_Generalization
export OPENAI_API_KEY=dummy
export QWEN_API_BASE="http://${VLLM_NODE}:31606/v1"
export OPENAI_API_BASE="$QWEN_API_BASE"

/home/mohanz/Harness_Generalization/SWE-bench/.venv/bin/python \
  /home/mohanz/Harness_Generalization/meta-harness/swebench_lite/formal_orchestrator.py \
  --config /home/mohanz/Harness_Generalization/meta-harness/swebench_lite/config_qwen3_8b.toml \
  --run qwen3-8b-swe-lite-metaharness-20261006 \
  --iterations 10 \
  --evolution-tasks 100 \
  --heldout-tasks 100 \
  --workers 3 \
  --api-base "$QWEN_API_BASE"
```

The orchestrator reconstructs the frontier from completed `iteration-*/evaluation.json`
files and resumes incomplete tasks using `--resume` in its worker. Existing results are
under `swebench_lite/runs/qwen3-8b-swe-lite-metaharness-20261006`; do not change the
`--run` value.

If the checkout has no `SWE-bench/.venv`, recreate it as documented in
`RESUME_QWEN3_4B.md` before running this command. The repository does not store
model weights, vLLM environments, or run artifacts.
