#!/usr/bin/env python3
"""Create fixed, reproducible benchmark splits for the first pilot study."""

import argparse
import json
import random
from pathlib import Path
from typing import Optional


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def webshop_split(source: Path, output: Path, seed: int):
    goals = json.loads(source.read_text())
    if not isinstance(goals, list) or len(goals) < 200:
        raise ValueError(f"Expected at least 200 WebShop goals, found {len(goals)}")
    indexed = list(enumerate(goals))
    random.Random(seed).shuffle(indexed)
    evolution = indexed[:100]
    test = indexed[100:200]
    write_json(output / "webshop_evolution.json", {
        "benchmark": "webshop",
        "split": "evolution",
        "seed": seed,
        "task_count": len(evolution),
        "tasks": [{"task_id": i, "instruction": task} for i, task in evolution],
    })
    write_json(output / "webshop_test.json", {
        "benchmark": "webshop",
        "split": "held_out_test",
        "seed": seed,
        "task_count": len(test),
        "tasks": [{"task_id": i, "instruction": task} for i, task in test],
    })


def alfworld_split(data_root: Path, output: Path, manifest_dir: Optional[Path] = None):
    if manifest_dir:
        files = {
            "evolution": manifest_dir / "valid_seen_games.jsonl",
            "held_out_test": manifest_dir / "valid_unseen_games.jsonl",
        }
        payloads = {}
        for split, path in files.items():
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            payloads[split] = rows
        write_json(output / "alfworld_evolution.json", {
            "benchmark": "alfworld",
            "split": "valid_seen_evolution",
            "task_count": len(payloads["evolution"]),
            "source": "canonical ALFWorld manifest",
            "tasks": payloads["evolution"],
        })
        write_json(output / "alfworld_test.json", {
            "benchmark": "alfworld",
            "split": "valid_unseen_held_out_test",
            "task_count": len(payloads["held_out_test"]),
            "source": "canonical ALFWorld manifest",
            "tasks": payloads["held_out_test"],
        })
        return
    # The native AgentBench checkout provides the frozen protocol manifests.
    # Prefer them over counting raw game directories, whose layout can contain
    # multiple trials for one benchmark task.
    frozen = {
        "evolution": data_root / "seen_valid.json",
        "held_out_test": data_root / "new_std.json",
    }
    if all(path.exists() for path in frozen.values()):
        payloads = {}
        for split, path in frozen.items():
            raw = json.loads(path.read_text())
            rows = []
            for task_type, gamefiles in raw.items():
                for gamefile in gamefiles:
                    rows.append({"task_type": task_type, "metadata": {"gamefile": gamefile}})
            payloads[split] = rows
        write_json(output / "alfworld_evolution.json", {
            "benchmark": "alfworld", "split": "valid_seen_evolution",
            "task_count": len(payloads["evolution"]), "source": str(frozen["evolution"]),
            "tasks": payloads["evolution"],
        })
        write_json(output / "alfworld_test.json", {
            "benchmark": "alfworld", "split": "valid_unseen_held_out_test",
            "task_count": len(payloads["held_out_test"]), "source": str(frozen["held_out_test"]),
            "tasks": payloads["held_out_test"],
        })
        return
    base = data_root / "json_2.1.1"
    # ALFWorld stores one directory per game, with the task JSON nested below it.
    seen = sorted(p for p in (base / "valid_seen").iterdir() if p.is_dir())
    unseen = sorted(p for p in (base / "valid_unseen").iterdir() if p.is_dir())
    if not seen or not unseen:
        raise FileNotFoundError(
            f"ALFWorld data not found under {base}. Run alfworld-download first."
        )
    write_json(output / "alfworld_evolution.json", {
        "benchmark": "alfworld",
        "split": "valid_seen_evolution",
        "task_count": len(seen),
        "task_ids": [p.stem for p in seen],
    })
    write_json(output / "alfworld_test.json", {
        "benchmark": "alfworld",
        "split": "valid_unseen_held_out_test",
        "task_count": len(unseen),
        "task_ids": [p.stem for p in unseen],
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--webshop-goals", type=Path, required=True)
    parser.add_argument("--alfworld-data", type=Path, required=False)
    parser.add_argument("--alfworld-manifest", type=Path, required=False)
    parser.add_argument("--output", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    webshop_split(args.webshop_goals, args.output, args.seed)
    if args.alfworld_data or args.alfworld_manifest:
        alfworld_split(args.alfworld_data, args.output, args.alfworld_manifest)
    print(f"Wrote splits to {args.output}")


if __name__ == "__main__":
    main()
