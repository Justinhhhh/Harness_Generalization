---
name: meta-harness-alfworld
description: Propose one ALFWorld harness candidate from the complete search archive.
---

# ALFWorld Meta-Harness proposer

Produce exactly one new executable harness candidate for the requested iteration.
The outer orchestrator owns validation and evaluation; do not run AgentBench or
start model servers.

## Required workflow

1. Read only the immutable seed configuration and `base_harness.py` under the
   run's `inputs/` directory, plus the complete search archive named in the
   prompt. Do not inspect the outer-loop implementation or repository files.
2. Read `frontier_val.json`, `evolution_summary.jsonl`, prior candidate source,
   and at least eight diverse `runs.jsonl` trajectories when available. Start
   with balanced successful and failed baseline rows. Each `read_trajectory`
   result contains the complete compact action-observation chain; compare the
   first point where a failed run diverges from task-directed behavior against
   successful behavior on the same task family. Do not rely on aggregate scores.
3. State one falsifiable, behavior-level hypothesis grounded in those traces.
4. Identify the exact behavior the guidance should change and the trajectory
   evidence that makes the change falsifiable. If prior attempts failed, explain
   why their policies were inert or harmful and choose a materially different policy.
5. Implement one mechanism: a concise model-facing guidance prompt that helps
   retain the task goal, searched locations, inventory, completed subgoals, and
   current admissible actions. Do not write Python or rewrite emitted actions.
   The outer loop renders a fixed standalone `AgentHarness` that appends the
   submitted guidance to the existing system prompt.
6. Call `submit_candidate` with `guidance` plus `name`,
   `hypothesis`, `changes`, `expected_effect`, `failure_analysis`,
   `trigger_and_effect`, and `self_check`. Metadata must describe the guidance
   actually submitted, not an earlier draft. This tool renders, validates, and
   atomically records the AgentHarness and metadata. If validation fails, revise
   the guidance and call `submit_candidate` again.

The generated candidate must cause an observable model-facing history change in
the interface contract test and record it in `self.trace`.

## Interface

`AgentHarness` is a complete session wrapper, not an H2/H3/H4/H5 plugin. The
outer loop owns its fixed implementation. The submitted guidance is appended to
the existing system prompt on every model turn; the underlying history and
emitted `take_action` tool call remain unchanged.

Before submission, check that the guidance is actionable from the task and
trajectory history alone, applies across every task family, and asks the model
to choose only from the latest AVAILABLE ACTIONS.

The base model is fixed. Keep the candidate general across ALFWorld tasks.
Do not mention task indices, game paths, held-out examples, object/location
combinations copied from individual tasks, or evaluation answers in code or
prompts. In particular, never copy numbered entities such as `sidetable 1`
from a trajectory into the candidate. Do not read or modify ALFWorld split manifests or held-out results.
Do not modify files outside the current iteration workspace.

Use the full filesystem history to diagnose behavior. One mechanism per
candidate; formatting-only changes and copies of the no-op seed are invalid.
