# Harness integration

This directory is the shared experiment boundary for Life-Harness and Meta-Harness.
The benchmark split files under `../data/splits` are authoritative; neither
upstream repository may silently replace them with its own task pool.

## Environment

```bash
export ALFWORLD_DATA=/srv/scratch/z5777347/alfworld-data
export HARNESS_ENV=/srv/scratch/z5777347/harness-env
export LIFE_HARNESS=/home/z5777347/Life-Harness
export META_HARNESS=/home/z5777347/metaharness
```

Run the deterministic checks first:

```bash
python3 scripts/verify_setup.py
```

## Evaluation contract

```text
alfworld_evolution.json → evolution/search only (140 valid_seen tasks)
alfworld_test.json       → final held-out test only (134 valid_unseen tasks)
webshop_evolution.json   → evolution/search only (100 tasks)
webshop_test.json        → final held-out test only (100 disjoint tasks)
```

The test split must not be passed to an evolution proposer. Model endpoint
credentials are supplied through environment variables and are never stored in
the split files or this repository.

## Upstream roles

- Life-Harness is the structured runtime baseline (H2/H3/H4/H5).
- Meta-Harness is the end-to-end proposer/search baseline.
- This repository owns the benchmark adapters and split contract.

The first executable milestone is an ALFWorld one-task smoke run. A full
evolution run requires a configured OpenAI-compatible model endpoint:

```bash
export AGENT_BASE_URL=http://localhost:30001/v1
export AGENT_MODEL=Qwen/Qwen3-4B-Instruct
export AGENT_API_KEY=EMPTY
```
