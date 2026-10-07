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

The complete trajectory bundle for this run is tracked in Git LFS at
`repro_artifacts/qwen3-8b-swe-lite-metaharness-20261006.tar.zst`. After
cloning and installing Git LFS, restore it with:

```bash
cd /home/mohanz/Harness_Generalization
git lfs pull --include='repro_artifacts/qwen3-8b-swe-lite-metaharness-20261006.tar.zst'
tar --zstd -xf repro_artifacts/qwen3-8b-swe-lite-metaharness-20261006.tar.zst \
  -C meta-harness/swebench_lite/runs
```

The archive restores the exact `runs/qwen3-8b-swe-lite-metaharness-20261006`
tree, including the partial evolution trajectories and evaluation state.

## Trajectory and resume contract

The run artifacts are the execution record and must be restored together for
an exact resume:

```text
baseline/trajectories/<id>/<id>.traj.json
iteration-XX/trajectories/<id>/<id>.traj.json
iteration-XX/trajectories/preds.json
iteration-XX/evaluation.json
iteration-XX/candidate-XX.yaml
iteration-XX/candidate_manifest.json
iteration-XX/proposer_trace.json
final/trajectories/<id>/<id>.traj.json
final/trajectories/preds.json
final/summary.json
```

Each task trajectory records the agent's observations, actions, tool calls,
and result. The proposer sees the previous evolution iteration's feedback and
selected evolution trajectories only. Heldout data is evaluated after the
evolution loop and is not exposed to the proposer. On resume, the worker uses
existing `preds.json` and task trajectory files to skip completed tasks and the
orchestrator uses completed `evaluation.json` files to rebuild the frontier.
If the run directory is absent, the Git checkout still reproduces the same
pipeline, but it cannot reproduce the prior partial state or exact prior
trajectories.
