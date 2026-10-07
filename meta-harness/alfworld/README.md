# ALFWorld Meta-Harness

This directory contains the ALFWorld-specific implementation of the
Meta-Harness propose/evaluate/archive loop. AgentBench remains the canonical
episode executor; this directory owns the proposer, candidate source, search
history, frontier, and final held-out result.

## Formal protocol

- Each run uses the same frozen model for trajectory execution and proposing:
  Qwen3-4B (`evolution.toml`) or Qwen3-8B (`evolution_8b.toml`) on a
  32,768-token vLLM endpoint. Trajectory requests use `max_tokens=2048`, while
  coding-agent proposer requests use `max_tokens=8192`; these are deliberately
  separate request budgets.
- Seed: `base_harness.py`, an independent no-op Meta-Harness candidate.
- Search: 10 coding-agent proposal iterations, each scored on all 140
  `alfworld_evolution` tasks with at most 50 environment steps.
- Selection: best evolution pass rate, with all candidates retained in the
  searchable filesystem archive.
- Final evaluation: the frozen winner on all 134 `alfworld_heldout` tasks.
- The proposer cannot inspect held-out results during search.

`ALFWORLD_HARNESS_FILE` is the sole candidate activation signal. When it is
present, AgentBench loads that standalone `AgentHarness` session wrapper. The
legacy `enabled/h2/.../h5` fields remain confined to raw-baseline and old
Life-Harness profiles; neither switch family is part of a candidate. Life-Harness
provides only the underlying AgentBench agent/session implementation.

## Completion contract

A run is complete only when all of the following exist and agree:

1. `protocol.json` records the fixed model, task counts, and iteration count.
2. `baseline/evolution` contains exactly the requested number of complete
   trajectories.
3. Every proposed candidate has proposer session logs, standalone source,
   interface-validation status, a one-task runtime smoke, and a complete
   evolution evaluation or an explicit failure record.
4. `frontier_val.json` identifies the best evaluated source.
5. `final/heldout` contains exactly 134 trajectories for that frozen source,
   and `final/summary.json` records its held-out score.
6. Every rollout contains an `ALFWORLD_HARNESS_AUDIT_V1` marker proving the
   exact candidate path was loaded; an exit code or aggregate score alone is
   insufficient.

The proposer is an OpenAI-compatible Qwen tool loop using the same frozen model
as that run's trajectory executor, not a one-shot text completion. Before a candidate is accepted, it must use `read_trajectory` to
inspect at least one successful and one failed raw trajectory. It may list,
search, and read the complete accumulated run archive, but may write only the
current `candidate-*.py` and `pending_eval.json`.

Each evolution iteration allows at most three proposer attempts. A failed
validation or runtime smoke is archived with its exact error and fed into the
next attempt so the proposer repairs the harness instead of repeating it. Only
a candidate that passes validation and smoke reaches the 140-task evolution
evaluation; three failed attempts produce an explicit `attempts_exhausted`
iteration record.

## Bring-up

Run the bounded end-to-end debug job first. It exercises baseline evaluation,
one coding-agent proposal, interface validation, candidate smoke, search-set
evaluation, frontier update, and held-out evaluation on tiny task slices:

```bash
sbatch jobs/pipeline_debug.sbatch
```

The isolated 8B smoke uses `jobs/pipeline_debug_8b.sbatch`; its run roots and
logs include `qwen3-8b` and never share frontier state with the 4B run.

After its `final/summary.json` and audit markers pass, submit the formal run:

```bash
sbatch jobs/pipeline_formal.sbatch
```

The orchestrator refuses to reuse an existing run root unless `--resume` is
explicit, and resume requires an identical `protocol.json`.

The debug batch job deliberately keeps its vLLM process and GPU allocation
alive after the first pipeline attempt. This permits corrected retries through
`srun --jobid=<job-id> --overlap ...` without releasing and reacquiring a GPU.
