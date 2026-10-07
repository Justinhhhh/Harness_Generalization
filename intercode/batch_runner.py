"""Run InterCode Bash tasks through the mini-swe-agent adapter.

Each task gets its own trajectory and filesystem setup selection.  This runner
does not inject gold commands into the model prompt; gold remains evaluator-only.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/nl2bash"
MINI = Path("/home/mohanz/Harness_Generalization/SWE-bench/.venv/bin/mini")
CONFIGS = {"qwen3-4b": ROOT / "minisweagent_qwen4b.yaml", "qwen3-8b": ROOT / "minisweagent_qwen8b.yaml"}
REPLAY_LOCK = threading.Lock()
REPLAY_LOCK_PATH = "/tmp/intercode-bash-replay.lock"


def remove_containers(*names: str) -> None:
    try:
        subprocess.run(["docker", "rm", "-f", *names], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=False, timeout=30)
    except subprocess.TimeoutExpired:
        pass


def tasks():
    for fs in range(1, 5):
        path = DATA / f"nl2bash_fs_{fs}.json"
        for index, record in enumerate(json.loads(path.read_text())):
            yield fs, index, record["query"], record["gold"]


def replay_reward(trajectory: dict, gold: str, setup: Path, model: str) -> tuple[float, dict]:
    """Replay the agent's commands in a fresh InterCode evaluator container.

    The mini-SWE trajectory only records observations and exit status; reward is
    recomputed against the dataset gold command using the canonical BashEnv
    evaluator, including filesystem and answer-similarity components.
    """
    from intercode.envs import BashEnv
    # BashEnv uses fixed container names; serialize replay so workers cannot
    # attach to or reset another task's evaluator containers.
    with REPLAY_LOCK, open(REPLAY_LOCK_PATH, "w") as lock_file:
      fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
      # Replay containers are dedicated to reward evaluation. Recreate them
      # for every task so root-level files from an earlier setup cannot leak.
      image_name = f"intercode-bash-replay-{model}"
      remove_containers(f"{image_name}_ic_ctr", f"{image_name}_ic_ctr_eval")
      # The prepared local image is the canonical InterCode Bash evaluator image.
      env = BashEnv(image_name, network_mode="none", verbose=False)
      try:
          env.reset()
          encoded = base64.b64encode(setup.read_bytes()).decode()
          for container in (env.container, env.container_eval):
              result = container.exec_run(["/bin/sh", "-c", f"echo {encoded} | base64 -d | /bin/sh"])
              if result.exit_code != 0:
                  raise RuntimeError(f"filesystem setup failed: {result.output!r}")
              baseline = container.exec_run([
                  "/bin/sh", "-c",
                  "git add -A && git -c user.name=InterCode -c user.email=intercode@local "
                  "commit --allow-empty -m task-baseline >/dev/null",
              ])
              if baseline.exit_code != 0:
                  raise RuntimeError(f"failed to establish task baseline: {baseline.output!r}")
          env.gold = gold
          env.observation = ""
          env.info = {}
          env.trajectory = []
          steps = trajectory.get("steps", trajectory.get("trajectory", []))
          if not steps:
              steps = [action for message in trajectory.get("messages", []) for action in message.get("extra", {}).get("actions", [])]
          for step in steps:
              action = (step.get("action") or step.get("command", "")) if isinstance(step, dict) else step[0]
              if not action or "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in action:
                  continue
              env.exec_action(action)
              env.trajectory.append((action, env.observation))
          reward, info = env.get_reward()
          return float(reward), info
      finally:
          remove_containers(env.container_name, env.ctr_name_eval)
          fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=CONFIGS, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--task-timeout", type=int, default=600)
    ap.add_argument("--replay-timeout", type=int, default=180)
    args = ap.parse_args()
    out = args.output or ROOT / "batch-results" / args.model
    out.mkdir(parents=True, exist_ok=True)
    all_rows = list(enumerate(tasks()))
    rows = all_rows[args.offset : args.offset + args.limit if args.limit else None]
    if args.resume:
        rows = [(n, row) for n, row in rows
                if not (out / f"task-{n:03d}.reward.json").exists()]
    def run_one(item):
        n, (fs, index, query, gold) = item
        # The setup path is selected per task; it is not included in the prompt.
        setup = ROOT / f"docker/bash_scripts/setup_nl2b_fs_{fs}.sh"
        cfg = out / f"config-{n:03d}.yaml"
        text = CONFIGS[args.model].read_text()
        text = text.replace("setup_nl2b_fs_1.sh", setup.name)
        text = text.replace("image_name: intercode-bash", f"image_name: intercode-bash-{args.model}")
        text += f"  container_suffix: task-{n:03d}\n"
        cfg.write_text(text)
        traj = out / f"task-{n:03d}.json"
        env = os.environ | {"MSWEA_CONFIGURED": "1", "PYTHONPATH": str(ROOT)}
        command = [str(MINI), "-c", str(cfg), "-t", query, "-y", "--exit-immediately", "-o", str(traj)]
        agent_image = f"intercode-bash-{args.model}"
        agent_containers = [
            f"{agent_image}_ic_ctr_task-{n:03d}",
            f"{agent_image}_ic_ctr_eval_task-{n:03d}",
        ]
        result = None
        if not (args.resume and traj.exists()):
            try:
                result = subprocess.run(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        timeout=args.task_timeout)
            except subprocess.TimeoutExpired as exc:
                output = exc.stdout or ""
                (out / f"task-{n:03d}.log").write_text(output)
                return {"index": n, "filesystem": fs, "query": query, "returncode": None,
                        "exit_status": "runner_timeout", "success": False, "reward": None,
                        "reward_info": None, "reward_error": "task timeout", "trajectory": str(traj)}
            finally:
                remove_containers(*agent_containers)
            (out / f"task-{n:03d}.log").write_text(result.stdout)
        exit_status = "missing"
        if traj.exists():
            try:
                trajectory = json.loads(traj.read_text())
                exit_status = trajectory.get("info", {}).get("exit_status", "missing")
            except Exception as exc:
                exit_status = f"trajectory_parse_error:{type(exc).__name__}"
                trajectory = None
        reward = None
        reward_info = None
        reward_error = None
        if trajectory is not None:
            reward_path = out / f"task-{n:03d}.reward.json"
            try:
                replay = subprocess.run(
                    [sys.executable, str(ROOT / "replay_worker.py"),
                     "--trajectory", str(traj), "--filesystem", str(fs),
                     "--dataset-index", str(index), "--model", args.model,
                     "--output", str(reward_path)],
                    cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    timeout=args.replay_timeout,
                )
                if replay.returncode != 0:
                    raise RuntimeError(replay.stdout[-2000:])
                replay_data = json.loads(reward_path.read_text())
                reward, reward_info = replay_data["reward"], replay_data["info"]
            except subprocess.TimeoutExpired:
                reward_error = f"reward replay timeout after {args.replay_timeout}s"
                remove_containers(f"intercode-bash-replay-{args.model}_ic_ctr",
                                  f"intercode-bash-replay-{args.model}_ic_ctr_eval")
                reward_path.write_text(json.dumps({
                    "reward": None, "info": {}, "reward_error": reward_error,
                }, indent=2))
            except Exception as exc:
                reward_error = f"{type(exc).__name__}: {exc}"
                reward_path.write_text(json.dumps({
                    "reward": None, "info": {}, "reward_error": reward_error,
                }, indent=2))
        return {
            "index": n,
            "filesystem": fs,
            "query": query,
            "returncode": result.returncode if result is not None else 0,
            "exit_status": exit_status,
            "success": exit_status == "Submitted",
            "reward": reward,
            "reward_info": reward_info,
            "reward_error": reward_error,
            "trajectory": str(traj),
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, item) for item in rows]
        summary = []
        for f in as_completed(futures):
            try:
                summary.append(f.result())
            except Exception as exc:
                summary.append({"error": f"{type(exc).__name__}: {exc}"})
    summary.sort(key=lambda x: x.get("index", 10**9))
    (out / "summary.partial.json").write_text(json.dumps(summary, indent=2))
    complete_summary = []
    for n, (fs, _, query, _) in all_rows:
        traj_path = out / f"task-{n:03d}.json"
        reward_path = out / f"task-{n:03d}.reward.json"
        if not (traj_path.exists() and reward_path.exists()):
            continue
        trajectory = json.loads(traj_path.read_text())
        reward_data = json.loads(reward_path.read_text())
        exit_status = trajectory.get("info", {}).get("exit_status", "missing")
        complete_summary.append({
            "index": n, "filesystem": fs, "query": query,
            "exit_status": exit_status, "success": exit_status == "Submitted",
            "reward": reward_data.get("reward"), "reward_info": reward_data.get("info"),
            "reward_error": reward_data.get("reward_error"), "trajectory": str(traj_path),
        })
    (out / "summary.json").write_text(json.dumps(complete_summary, indent=2, default=float))


if __name__ == "__main__":
    main()
