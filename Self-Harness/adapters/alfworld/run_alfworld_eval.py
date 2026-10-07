#!/usr/bin/env python3
"""Run an ALFWorld candidate through AgentBench and emit Self-Harness results."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

META_ROOT = Path("/home/mohanz/Harness_Generalization/meta-harness").resolve()
sys.path.insert(0, str(META_ROOT / "alfworld"))
from formal_orchestrator import evaluate  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--evolution-tasks", type=int, default=int(os.environ.get("SELF_HARNESS_ALFWORLD_EVOLUTION_TASKS", "140")))
    p.add_argument("--heldout-tasks", type=int, default=int(os.environ.get("SELF_HARNESS_ALFWORLD_HELDOUT_TASKS", "134")))
    p.add_argument("--api-base", default=os.environ.get("QWEN_API_BASE", "http://unites4.ib:30017/v1"))
    p.add_argument("--model", default=os.environ.get("QWEN_MODEL", "qwen3-4b"))
    p.add_argument("--agent-name", default=os.environ.get("QWEN_AGENT_NAME", "qwen3-4b"))
    p.add_argument("--agent-max-tokens", type=int, default=2048)
    p.add_argument(
        "--splits",
        default=os.environ.get("SELF_HARNESS_ALFWORLD_SPLITS", "evolution,heldout"),
        help="Comma-separated splits to run: evolution and/or heldout.",
    )
    args = p.parse_args()
    candidate = args.candidate.expanduser().resolve()
    if candidate.is_dir():
        for name in ("AgentHarness.py", "agent_harness.py", "alfworld_harness_surface.py", "base_harness.py"):
            for root in (candidate / "current", candidate):
                if (root / name).is_file():
                    candidate = root / name
                    break
            if candidate.is_file():
                break
    if not candidate.is_file():
        raise SystemExit(f"candidate harness not found: {candidate}")
    os.environ.update({"QWEN_API_BASE": args.api_base, "QWEN_MODEL": args.model, "QWEN_AGENT_NAME": args.agent_name, "CHILD_MAX_TOKENS": str(args.agent_max_tokens)})
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    splits = {}
    requested = {item.strip() for item in args.splits.split(",") if item.strip()}
    unknown = requested - {"evolution", "heldout"}
    if unknown:
        raise SystemExit(f"unknown ALFWorld split(s): {sorted(unknown)}")
    for split, phase, expected in (("evolution", "candidate-evolution", args.evolution_tasks), ("heldout", "heldout", args.heldout_tasks)):
        if split not in requested:
            continue
        result = evaluate(candidate, phase, out / split, expected, os.getpid() + expected)
        splits[split] = [{"repeat": 1, "passed": int(result["pass"]), "total": int(result["total"]), "pass_rate": float(result["pass_rate"])}]
    payload = {"format": "self_harness.alfworld_result.v0", "candidate": str(candidate), "splits": splits}
    (out / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
