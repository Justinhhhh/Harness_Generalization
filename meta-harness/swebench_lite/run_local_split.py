#!/usr/bin/env python3
"""Run mini-SWE-agent on a local SWE-bench JSONL split."""
from __future__ import annotations
import argparse, concurrent.futures, json, threading, time, traceback
from pathlib import Path
from minisweagent.config import get_config_from_spec
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import get_sb_environment, update_preds_file
from agent_harness import BudgetedSWEAgent
from minisweagent.utils.serialize import recursive_merge

class Progress:
    def on_instance_start(self, *a): pass
    def on_instance_end(self, *a): pass
    def update_instance_status(self, *a): pass

def one(instance, output, config):
    iid = instance["instance_id"]; directory = output / iid; directory.mkdir(parents=True, exist_ok=True)
    agent = None; status = ""; submission = ""
    for attempt in range(3):
        try:
            env = get_sb_environment(config, instance)
            agent = BudgetedSWEAgent(get_model(config=config.get("model", {})), env, progress_manager=Progress(), instance_id=iid, **config.get("agent", {}))
            info = agent.run(instance["problem_statement"]); status = info.get("exit_status", ""); submission = info.get("submission", "")
            break
        except Exception as exc:
            status = type(exc).__name__
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            (directory / f"{iid}.error.json").write_text(json.dumps(error, indent=2))
            print(f"[{iid}] attempt={attempt + 1} {type(exc).__name__}: {exc}", flush=True)
            if "Connection error" not in str(exc) or attempt == 2:
                break
            time.sleep(2 ** attempt)
    if agent is not None:
        agent.save(directory / f"{iid}.traj.json", {"info": {"exit_status": status, "submission": submission}, "instance_id": iid})
    update_preds_file(output / "preds.json", iid, config["model"]["model_name"], submission)


def is_retryable_trajectory(path):
    try:
        return json.loads(path.read_text()).get("info", {}).get("exit_status") == "InternalServerError"
    except (OSError, json.JSONDecodeError):
        return True

def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--model", required=True); p.add_argument("--workers", type=int, default=1); p.add_argument("--config", action="append", required=True); p.add_argument("--resume", action="store_true"); a=p.parse_args()
    config=recursive_merge(*(get_config_from_spec(x) for x in a.config), {"model": {"model_name": a.model}}); a.output.mkdir(parents=True, exist_ok=True)
    items=[json.loads(x) for x in a.dataset.read_text().splitlines() if x.strip()]
    if a.resume:
        items = [item for item in items if not ((path := a.output / item["instance_id"] / f'{item["instance_id"]}.traj.json').exists() and not is_retryable_trajectory(path))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex: list(ex.map(lambda x: one(x,a.output,config),items))
if __name__ == "__main__": main()
