#!/usr/bin/env python3
"""Create a compact train-side diagnosis brief from ALFWorld runs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def rollout_summary(row: dict) -> dict[str, object] | None:
    """Extract a compact, proposer-safe summary from one native rollout."""
    result = ((row.get("output") or {}).get("result") or {})
    messages = result.get("openai_messages") or []
    if not isinstance(messages, list):
        return None
    task = ""
    actions: list[str] = []
    observations: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = str(message.get("content") or "")
        if role == "user" and "Your task is to:" in content:
            match = re.search(r"Your task is to:\s*(.+?)(?:\s+AVAILABLE ACTIONS:|$)", content, re.S)
            if match:
                task = " ".join(match.group(1).split())
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                try:
                    arguments = json.loads(call["function"]["arguments"])
                    action = arguments.get("action")
                    if isinstance(action, str):
                        actions.append(action)
                except (KeyError, TypeError, json.JSONDecodeError):
                    continue
        if role == "tool":
            observation = content.split(" AVAILABLE ACTIONS:", 1)[0].strip()
            if observation:
                observations.append(observation)
    return {
        "index": row.get("index"),
        "reward": float(result.get("reward") or 0),
        "task": task or "unavailable",
        "actions": actions[-8:],
        "observations": observations[-4:],
    }


def render_case(label: str, summary: dict[str, object]) -> str:
    actions = summary["actions"] or ["no tool action recorded"]
    observations = summary["observations"] or ["no environment observation recorded"]
    return (
        f"### {label} evolution case {summary['index']}\n"
        f"Task: {summary['task']}\n"
        f"Recent actions: {' | '.join(actions)}\n"
        f"Recent observations: {' | '.join(observations)}\n"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    rows = []
    for path in (args.work_dir / "baseline_eval").rglob("runs.jsonl"):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    summaries = [summary for row in rows if (summary := rollout_summary(row)) is not None]
    passing = [item for item in summaries if item["reward"] > 0]
    failing = [item for item in summaries if item["reward"] <= 0]
    passing.sort(key=lambda item: int(item["index"]) if isinstance(item["index"], int) else -1)
    failing.sort(key=lambda item: int(item["index"]) if isinstance(item["index"], int) else -1)
    failed_cases = "\n".join(render_case("Unresolved", item) for item in failing[:4])
    passed_cases = "\n".join(render_case("Successful", item) for item in passing[:2])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "# ALFWorld train-side diagnosis\n\n"
        f"Observed {len(summaries)} evolution trajectories. Rewards > 0: {len(passing)}.\n\n"
        "The following are representative evolution-only action/observation traces. "
        "Use them to identify a reusable prompt-level behavior; preserve successful behavior.\n\n"
        "## Unresolved trajectory evidence\n"
        f"{failed_cases or 'No unresolved trajectory was recorded.'}\n"
        "## Successful trajectory evidence\n"
        f"{passed_cases or 'No successful trajectory was recorded.'}\n"
        "Propose one reusable prompt-level harness change. Preserve exact available action strings, "
        "emit one take_action call per turn, and do not rewrite actions.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
