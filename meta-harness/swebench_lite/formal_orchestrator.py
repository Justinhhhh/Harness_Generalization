#!/usr/bin/env python3
"""Meta-Harness evolution loop for SWE-bench Lite prompt harnesses.

Candidates alter only the mini-SWE-agent system prompt.  The executor model,
task split, Docker image, and test harness stay frozen.  Search uses only the
evolution split; the eval split is touched only once after a winner is frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
import urllib.request
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SWE = PROJECT / "SWE-bench"
PYTHON = SWE / ".venv" / "bin" / "python"
DEFAULT = (SWE / ".venv" / "lib" / "python3.11" / "site-packages" /
           "minisweagent" / "config" / "benchmarks" / "swebench.yaml")


def rows(path: Path, limit: int) -> list[dict]:
    data = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return data[:limit] if limit else data


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


BASE_SYSTEM_TEMPLATE = """You are a helpful assistant that can interact with a computer shell to solve programming tasks.
Work in a bounded budget. Investigate briefly, then implement a minimal source-only fix in /testbed.
Use the task instructions as the authority for investigation, editing, testing, and submission.
Work naturally: inspect enough context to understand the bug, make a minimal source-only fix,
verify it when useful, and follow the exact submission protocol from the task instructions when
the solution is ready. Do not commit or modify tests/configuration unless the task requires it.
The shell is non-interactive: do not use editors or sudo. After a nonzero command result, inspect
the error and change the command rather than repeating it. Quote edits safely, inspect git diff
before submission, and submit a minimal patch once it is ready."""


def system_template_from_config(path: Path) -> str:
    """Read the rendered system prompt without requiring PyYAML."""
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines)
                 if line.strip() == "system_template: |") + 1
    prompt = []
    for line in lines[start:]:
        if line.startswith("  ") and not line.startswith("    "):
            break
        prompt.append(line[4:] if line.startswith("    ") else line)
    return "\n".join(prompt).rstrip()


def candidate_config(path: Path, guidance: str, step_limit: int, max_tokens: int,
                     parent: Path | None = None) -> str:
    # Match MetaHarness frontier semantics: a new candidate is rendered from the
    # current best candidate, then the proposer adds a general delta.  The
    # executor, environment, and model settings remain frozen.
    parent_prompt = system_template_from_config(parent) if parent else BASE_SYSTEM_TEMPLATE
    additions = [part.strip() for part in (parent_prompt, guidance.strip()) if part.strip()]
    rendered_prompt = "\n".join(dict.fromkeys(additions))
    indented = textwrap.indent(rendered_prompt, "    ")
    text = """agent:
  system_template: |
{guidance}
  step_limit: {step_limit}
  max_consecutive_format_errors: 8
model:
  cost_tracking: ignore_errors
  model_kwargs:
    max_tokens: {max_tokens}
    chat_template_kwargs:
      enable_thinking: false
""".format(guidance=indented, step_limit=step_limit, max_tokens=max_tokens)
    path.write_text(text)
    return hashlib.sha256(rendered_prompt.encode()).hexdigest()


def run_agent(dataset: Path, candidate: Path, output: Path, model: str, workers: int, resume: bool = True) -> None:
    """Run mini-SWE-agent directly against a local JSONL split."""
    # mini's CLI expects a HuggingFace dataset identifier, whereas the experiment
    # splits are JSONL.  The local driver keeps the official agent/environment but
    # supplies the parsed records directly.
    driver = ROOT / "run_local_split.py"
    cmd = [str(PYTHON), str(driver), "--dataset", str(dataset), "--output", str(output),
           "--model", model, "--workers", str(workers), "--config", str(DEFAULT),
           "--config", str(candidate)]
    if resume:
        cmd.append("--resume")
    subprocess.run(cmd, cwd=SWE, check=True)


def score(dataset: Path, predictions: Path, run_id: str, workers: int) -> dict:
    """Official SWE-bench test evaluation; returns the generated report summary."""
    cmd = [str(PYTHON), "-m", "swebench.harness.run_evaluation", "--dataset_name", str(dataset),
           "--split", "test", "--predictions_path", str(predictions), "--run_id", run_id,
           "--max_workers", str(workers), "--timeout", "1800"]
    subprocess.run(cmd, cwd=SWE, check=True)
    report = SWE / "logs" / "evaluation" / run_id / "results.json"
    if not report.exists():
        raise RuntimeError(f"no report produced for {run_id}")
    return json.loads(report.read_text())


def resolved_rate(report: dict) -> float:
    resolved = report.get("resolved_ids") or []
    total = report.get("total_instances") or report.get("total") or 0
    return len(resolved) / total if total else 0.0


def recorded_score(path: Path) -> float:
    """Read a completed candidate score for frontier reconstruction on resume."""
    try:
        record = json.loads(path.read_text())
        if isinstance(record.get("score"), (int, float)):
            return float(record["score"])
        return resolved_rate(record.get("report") or {})
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return -1.0


def reconstruct_frontier(runroot: Path, baseline_cfg: Path) -> tuple[Path, float]:
    """Recover the best candidate after a process/GPU restart."""
    best_cfg, best_score = baseline_cfg, -1.0
    baseline_eval = runroot / "baseline" / "evaluation.json"
    if baseline_eval.exists():
        best_score = recorded_score(baseline_eval)
    for workspace in sorted(runroot.glob("iteration-[0-9][0-9]")):
        candidate = workspace / f"candidate-{workspace.name[-2:]}.yaml"
        evaluation = workspace / "evaluation.json"
        score_value = recorded_score(evaluation) if evaluation.exists() else -1.0
        if candidate.exists() and score_value > best_score:
            best_cfg, best_score = candidate, score_value
    return best_cfg, best_score


def smoke_passed(report: dict) -> bool:
    """A smoke gate checks execution integrity, not task correctness.

    A one-task smoke may legitimately produce an unresolved or empty patch for a
    small model. It must, however, finish without infrastructure/model/grader
    errors and produce exactly one submitted trajectory record.
    """
    total = report.get("total_instances") or report.get("total") or 0
    return bool(total and report.get("submitted_instances") == total
                and not report.get("error_instances")
                and not report.get("infra_failure_instances"))


def candidate_loaded(candidate: Path, trajectory_root: Path, guidance: str = "") -> bool:
    """Verify the smoke agent actually received the candidate system prompt."""
    if not guidance:
        return True  # baseline has no proposer guidance
    # The overlay is serialized into mini-SWE-agent's trajectory config. Checking
    # the exact structured submission avoids mistaking a baseline run for a run
    # with the new harness.
    for path in trajectory_root.glob("*/*.traj.json"):
        if guidance in path.read_text():
            return True
    return False


def _proposer_call(api_base: str, payload: dict) -> dict:
    req = urllib.request.Request(api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.loads(response.read())


def _safe_trajectory(root: Path, instance_id: str) -> Path | None:
    """Resolve only the known trajectory filename; proposer never gets arbitrary paths."""
    if not instance_id or Path(instance_id).name != instance_id:
        return None
    matches = list(root.glob(f"*/{instance_id}.traj.json"))
    return matches[0] if len(matches) == 1 else None


def _trajectory_for_proposer(path: Path) -> dict:
    item = json.loads(path.read_text())
    messages = item.get("messages") or []
    # Keep the raw interaction available, but bound each message and the whole tool result.
    clipped = []
    for message in messages[-8:]:
        if not isinstance(message, dict):
            continue
        out = {"role": message.get("role", "")}
        if isinstance(message.get("content"), str):
            out["content"] = message["content"][-700:]
        if message.get("tool_calls"):
            out["tool_calls"] = message["tool_calls"]
        clipped.append(out)
    info = item.get("info") or {}
    return {"instance_id": item.get("instance_id"), "exit_status": info.get("exit_status"),
            "submission": bool(info.get("submission")), "messages": clipped}


def _compact_feedback(items: list[dict]) -> list[dict]:
    """Keep proposer observations bounded by the 32k executor context window."""
    compact = []
    for item in items[:20]:
        excerpts = item.get("excerpt") or []
        last = excerpts[-1] if excerpts else {}
        compact.append({"instance_id": item.get("instance_id"),
                        "grader_status": item.get("grader_status"),
                        "exit_status": item.get("exit_status"),
                        "calls": item.get("calls"),
                        "reward": item.get("reward"),
                        "last_observation": {"role": last.get("role"),
                                             "content": str(last.get("content", ""))[-600:]}})
    return compact


def propose(api_base: str, model: str, baseline: float, feedback: dict, max_tokens: int,
            trajectory_root: Path, workspace: Path, iteration: int,
            parent: Path, parent_score: float) -> dict:
    """Run the proposer as a structured, auditable tool interaction.

    The proposer may inspect feedback and selected raw trajectories through tools. It
    cannot read arbitrary files or modify the harness directly; the only write is a
    validated submit_candidate result consumed by this orchestrator.
    """
    tools = [
        {"type": "function", "function": {"name": "read_feedback",
         "description": "Read the current official score and failure distribution.",
         "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
        {"type": "function", "function": {"name": "read_trajectory",
         "description": "Inspect one completed evolution trajectory by instance id.",
         "parameters": {"type": "object", "properties": {"instance_id": {"type": "string"}},
                        "required": ["instance_id"], "additionalProperties": False}}},
        {"type": "function", "function": {"name": "submit_candidate",
         "description": "Submit one general prompt addition after observing feedback and a trajectory.",
         "parameters": {"type": "object", "properties": {
             "summary": {"type": "string"}, "guidance": {"type": "string"},
             "failure_analysis": {"type": "string"}, "self_check": {"type": "string"}},
             "required": ["summary", "guidance", "failure_analysis", "self_check"],
             "additionalProperties": False}}},
    ]
    report = feedback.get("report") or {}
    parent_prompt = system_template_from_config(parent)
    system = ("You are the proposer in a MetaHarness experiment for a fixed SWE-bench agent. "
              "You may only improve the agent by adding general system-prompt guidance. "
              "First call read_feedback and read_trajectory for at least one instance. Then "
              "call submit_candidate exactly once. Do not use task-specific answers, gold patches, "
              "hidden tests, instance ids, or arbitrary file access. Guidance must be one general "
              "workflow addition under 180 words. The executor, tools, environment, step limit, "
              "grader, and task split are frozen. Thinking is disabled for the executor. "
              "The candidate must preserve the current best harness behavior and add only "
              "the new general improvement.")
    available_ids = [p.name for p in sorted(trajectory_root.iterdir()) if p.is_dir()]
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "Iteration %d; current frontier score %.3f; proposer feedback score %.3f. "
                 "The current best system prompt is below. Preserve it; submit only a general delta. "
                 "Available instances: %s. Use the tools to inspect evidence, then submit a candidate.\n\n%s" %
                 (iteration, parent_score, baseline,
                  ", ".join(available_ids[:20]),
                  parent_prompt)}]
    trace = []
    read_feedback_done = False
    read_trajectory_done = False
    observed_code_entities: set[str] = set()
    submitted = None
    for _ in range(12):
        body = _proposer_call(api_base, {"model": model, "temperature": 0.2,
            "max_tokens": max_tokens, "messages": messages, "tools": tools,
            "tool_choice": "auto", "chat_template_kwargs": {"enable_thinking": False}})
        message = body.get("choices", [{}])[0].get("message", {})
        trace.append({"response": body})
        # vLLM adds nullable response fields (audio/refusal/reasoning/etc.).
        # Keep only the OpenAI tool-loop fields when feeding the next request;
        # this avoids parser-specific fields growing the context or being rejected.
        assistant_message = {"role": "assistant", "content": message.get("content")}
        if message.get("tool_calls"):
            assistant_message["tool_calls"] = message["tool_calls"]
        messages.append(assistant_message)
        calls = message.get("tool_calls") or []
        if not calls:
            messages.append({"role": "user", "content": "Use the required tools now; do not answer with prose."})
            continue
        for call in calls:
            name = ((call.get("function") or {}).get("name"))
            raw_args = ((call.get("function") or {}).get("arguments")) or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError as exc:
                result = {"ok": False, "error": f"invalid JSON arguments: {exc}"}
            else:
                if name == "read_feedback":
                    read_feedback_done = True
                    result = {"ok": True, "baseline": baseline, "report": report,
                              "summary": _compact_feedback(feedback.get("summary") or [])}
                elif name == "read_trajectory":
                    path = _safe_trajectory(trajectory_root, str(args.get("instance_id", "")))
                    if path is None:
                        result = {"ok": False, "error": "unknown or ambiguous instance_id"}
                    else:
                        read_trajectory_done = True
                        trajectory = _trajectory_for_proposer(path)
                        raw = json.dumps(trajectory, ensure_ascii=False)
                        observed_code_entities.update(re.findall(
                            r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b|\b[A-Z][a-z]+[A-Z][A-Za-z0-9]*\b", raw))
                        result = {"ok": True, "trajectory": trajectory}
                elif name == "submit_candidate":
                    guidance = str(args.get("guidance", "")).strip()
                    words = guidance.split()
                    forbidden = [x for x in ("gold patch", "hidden test", ".traj.json") if x in guidance.lower()]
                    leaked_entities = [x for x in observed_code_entities
                                       if len(x) > 5 and x.lower() in guidance.lower()]
                    if not read_feedback_done or not read_trajectory_done:
                        result = {"ok": False, "error": "read_feedback and read_trajectory are required first"}
                    elif not guidance or len(words) > 180 or forbidden or leaked_entities:
                        result = {"ok": False, "error": "guidance must be nonempty, <=180 words, and general",
                                  "task_specific_entities": leaked_entities[:20],
                                  "retry_instruction": "Rewrite from scratch using only general agent behavior; "
                                  "do not mention any function, class, file, library, benchmark instance, or code identifier."}
                    else:
                        submitted = {"summary": str(args.get("summary", "")),
                                     "guidance": guidance,
                                     "failure_analysis": str(args.get("failure_analysis", "")),
                                     "self_check": str(args.get("self_check", ""))}
                        result = {"ok": True, "accepted": True, "word_count": len(words)}
                else:
                    result = {"ok": False, "error": f"unknown tool {name}"}
            trace.append({"tool_call": call, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                             "content": json.dumps(result, ensure_ascii=False)})
        if submitted is not None:
            break
    if submitted is None:
        workspace.mkdir(parents=True, exist_ok=True)
        atomic_json(workspace / "proposer_trace.json", {"iteration": iteration, "messages": trace,
            "submitted": None, "read_feedback": read_feedback_done,
            "read_trajectory": read_trajectory_done})
        raise RuntimeError("proposer did not submit a valid structured candidate")
    workspace.mkdir(parents=True, exist_ok=True)
    atomic_json(workspace / "proposer_trace.json", {"iteration": iteration, "messages": trace,
        "submitted": submitted, "read_feedback": read_feedback_done,
        "read_trajectory": read_trajectory_done})
    return submitted


def trajectory_summary(output: Path, report: dict | None = None) -> list[dict]:
    report = report or {}
    resolved = set(report.get("resolved_ids") or [])
    empty = set(report.get("empty_patch_ids") or [])
    errors = set(report.get("error_ids") or []) | set(report.get("infra_failure_ids") or [])
    result = []
    for path in sorted(output.glob("*/*.traj.json")):
        item = json.loads(path.read_text())
        info = item.get("info", {})
        messages = item.get("messages") or []
        excerpt = []
        for message in messages[-10:]:
            if not isinstance(message, dict):
                continue
            role = message.get("role", "")
            content = message.get("content")
            if isinstance(content, str):
                excerpt.append({"role": role, "content": content[-1200:]})
            elif message.get("tool_calls"):
                excerpt.append({"role": role, "tool_calls": message["tool_calls"]})
        result.append({"instance_id": item.get("instance_id"), "exit_status": info.get("exit_status"),
                       "calls": (info.get("model_stats") or {}).get("api_calls"),
                       "submitted": bool(info.get("submission")),
                       "reward": 1.0 if item.get("instance_id") in resolved else 0.0,
                       "grader_status": ("resolved" if item.get("instance_id") in resolved
                                         else "empty_patch" if item.get("instance_id") in empty
                                         else "error" if item.get("instance_id") in errors
                                         else "unresolved"),
                       "excerpt": excerpt})
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=ROOT / "config.toml", type=Path)
    p.add_argument("--run", default=f"qwen3-4b-swe-lite-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    p.add_argument("--iterations", type=int)
    p.add_argument("--evolution-tasks", type=int)
    p.add_argument("--heldout-tasks", type=int)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--evolution-workers", type=int,
                   help="Concurrent task batch size for evolution evaluations; defaults to --workers.")
    p.add_argument("--api-base", default=os.environ.get("QWEN_API_BASE", ""))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    evolution_workers = args.evolution_workers or args.workers
    if evolution_workers < 1:
        raise SystemExit("--evolution-workers must be >= 1")
    cfg = tomllib.loads(args.config.read_text())
    runroot = ROOT / "runs" / args.run
    evolution = rows(Path(cfg["evolution_split"]), args.evolution_tasks or int(cfg.get("evolution_tasks", 10)))
    heldout = rows(Path(cfg["eval_split"]), args.heldout_tasks or int(cfg.get("heldout_tasks", 100)))
    iterations = args.iterations if args.iterations is not None else int(cfg["iterations"])
    model = str(cfg.get("executor_model", cfg["proposer_model"]))
    atomic_json(runroot / "protocol.json", {"model": model, "iterations": iterations,
        "evolution_ids": [x["instance_id"] for x in evolution], "heldout_ids": [x["instance_id"] for x in heldout]})
    if args.dry_run:
        print(json.dumps({"runroot": str(runroot), "evolution": len(evolution), "heldout": len(heldout),
                          "command": "ready"}, indent=2))
        return
    if not args.api_base:
        raise SystemExit("QWEN_API_BASE is required for proposer evolution")
    evo_path, eval_path = runroot / "evolution.jsonl", runroot / "heldout.jsonl"
    evo_path.write_text("\n".join(json.dumps(x) for x in evolution) + "\n")
    eval_path.write_text("\n".join(json.dumps(x) for x in heldout) + "\n")
    smoke_path = runroot / "smoke.jsonl"
    smoke_path.write_text(json.dumps(evolution[0]) + "\n")
    baseline_cfg = runroot / "baseline.yaml"
    if not baseline_cfg.exists():
        candidate_config(baseline_cfg, "", int(cfg.get("step_limit", 75)),
                         int(cfg.get("executor_max_tokens", 4096)))
    best_cfg, best_score = reconstruct_frontier(runroot, baseline_cfg)
    history: list[dict] = []
    completed_iterations = []
    for prior in sorted(runroot.glob("iteration-[0-9][0-9]/evaluation.json")):
        try:
            record = json.loads(prior.read_text())
            if isinstance(record.get("iteration"), int) and record.get("report"):
                history.append(record)
                completed_iterations.append(record["iteration"])
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    start_iteration = max(completed_iterations, default=0) + 1
    for iteration in range(start_iteration, iterations + 1):
        workspace = runroot / ("baseline" if iteration == 0 else f"iteration-{iteration:02d}")
        workspace.mkdir(parents=True, exist_ok=True)
        candidate = best_cfg if iteration == 0 else workspace / f"candidate-{iteration:02d}.yaml"
        if iteration:
            manifest = workspace / "candidate_manifest.json"
            manifest_data = json.loads(manifest.read_text()) if manifest.exists() else {}
            lineage_matches = manifest_data.get("parent") == str(best_cfg.resolve())
            if (candidate.exists() and manifest.exists() and lineage_matches
                    and (workspace / "trajectories" / "preds.json").exists()):
                # Resume a partially completed iteration without asking the proposer
                # to regenerate a potentially different candidate.
                guidance = manifest_data.get("guidance", "")
            else:
                if candidate.exists() or manifest.exists():
                    stale = workspace / f"stale-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
                    stale.mkdir(parents=True, exist_ok=True)
                    for name in ("candidate-" + f"{iteration:02d}" + ".yaml",
                                 "candidate_manifest.json", "proposal.txt",
                                 "proposer_trace.json", "trajectories", "smoke"):
                        old = workspace / name
                        if old.exists():
                            shutil.move(str(old), str(stale / name))
                proposal = propose(
                    args.api_base,
                    str(cfg["proposer_model"]),
                    best_score,
                    history[-1] if history else {"summary": []},
                    int(cfg.get("proposer_max_tokens", 8192)),
                    (runroot / ("baseline" if iteration == 1 else f"iteration-{iteration-1:02d}") / "trajectories"),
                    workspace, iteration, best_cfg, best_score,
                )
                guidance = proposal["guidance"]
                guidance_hash = candidate_config(
                    candidate, guidance, int(cfg.get("step_limit", 75)),
                    int(cfg.get("executor_max_tokens", 4096)), parent=best_cfg)
                atomic_json(workspace / "candidate_manifest.json", {
                    "source": "structured_proposer_submit_candidate",
                    "iteration": iteration, "candidate": str(candidate),
                    "parent": str(best_cfg.resolve()), "parent_score": best_score,
                    "guidance_sha256": guidance_hash, "guidance": guidance,
                    "executor_model": model, "thinking": False,
                    "step_limit": int(cfg.get("step_limit", 75)),
                    "executor_max_tokens": int(cfg.get("executor_max_tokens", 4096)),
                })
                (workspace / "proposal.txt").write_text(guidance + "\n")
        smoke_out = workspace / "smoke" / "trajectories"
        run_agent(smoke_path, candidate, smoke_out, model, args.workers)
        smoke_report = score(smoke_path, smoke_out / "preds.json", f"{args.run}-smoke-{iteration}", args.workers)
        loaded = candidate_loaded(candidate, smoke_out, guidance if iteration else "")
        if not smoke_passed(smoke_report) or not loaded:
            record = {"iteration": iteration, "candidate": str(candidate), "score": 0.0,
                      "smoke": {"passed": False, "loaded": loaded, "report": smoke_report},
                      "summary": trajectory_summary(smoke_out, smoke_report), "outcome": "smoke_failed"}
            history.append(record)
            atomic_json(workspace / "evaluation.json", record)
            if iteration == 0:
                raise RuntimeError("baseline smoke failed; refusing to run evolution")
            continue
        run_agent(evo_path, candidate, workspace / "trajectories", model, evolution_workers)
        report = score(evo_path, workspace / "trajectories" / "preds.json", f"{args.run}-evo-{iteration}", evolution_workers)
        value = resolved_rate(report)
        record = {"iteration": iteration, "candidate": str(candidate), "score": value,
                  "smoke": {"passed": True, "loaded": loaded, "report": smoke_report},
                  "report": report, "summary": trajectory_summary(workspace / "trajectories", report)}
        history.append(record)
        atomic_json(workspace / "evaluation.json", record)
        if value > best_score:
            best_cfg, best_score = candidate, value
    final = runroot / "final"
    run_agent(eval_path, best_cfg, final / "trajectories", model, args.workers)
    report = score(eval_path, final / "trajectories" / "preds.json", f"{args.run}-heldout", args.workers)
    atomic_json(final / "summary.json", {"winner": str(best_cfg), "evolution_score": best_score,
        "heldout_report": report, "heldout_score": resolved_rate(report)})


if __name__ == "__main__":
    main()
