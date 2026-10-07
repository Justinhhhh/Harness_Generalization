"""Run one reward replay in an isolated process."""
import argparse
import json
from pathlib import Path

from batch_runner import DATA, ROOT, replay_reward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--filesystem", type=int, required=True)
    parser.add_argument("--dataset-index", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = json.loads((DATA / f"nl2bash_fs_{args.filesystem}.json").read_text())
    gold = records[args.dataset_index]["gold"]
    setup = ROOT / f"docker/bash_scripts/setup_nl2b_fs_{args.filesystem}.sh"
    trajectory = json.loads(args.trajectory.read_text())
    reward, info = replay_reward(trajectory, gold, setup, args.model)
    args.output.write_text(json.dumps({"reward": reward, "info": info}, indent=2, default=float))


if __name__ == "__main__":
    main()
