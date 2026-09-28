#!/usr/bin/env python3
"""Verify data, task manifests, and upstream checkout paths without running a model."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"


def check_split(name, expected):
    path = SPLITS / name
    if not path.exists():
        raise SystemExit(f"missing split: {path}")
    data = json.loads(path.read_text())
    actual = data.get("task_count")
    if actual != expected:
        raise SystemExit(f"{name}: expected {expected}, found {actual}")
    return data


def main():
    evo = check_split("alfworld_evolution.json", 140)
    test = check_split("alfworld_test.json", 134)
    web_evo = check_split("webshop_evolution.json", 100)
    web_test = check_split("webshop_test.json", 100)

    if os.environ.get("ALFWORLD_DATA"):
        data_root = Path(os.environ["ALFWORLD_DATA"])
        missing = []
        for row in evo["tasks"] + test["tasks"]:
            gamefile = row["metadata"]["gamefile"].replace("${ALFWORLD_DATA}", str(data_root))
            if not Path(gamefile).exists():
                missing.append(gamefile)
        if missing:
            raise SystemExit(f"ALFWorld missing {len(missing)} game files")

    web_ids = {row["task_id"] for row in web_evo["tasks"]}
    web_test_ids = {row["task_id"] for row in web_test["tasks"]}
    if web_ids & web_test_ids:
        raise SystemExit("WebShop evolution/test task IDs overlap")

    for name in ("/home/z5777347/Life-Harness", "/home/z5777347/meta-harness"):
        if not Path(name).is_dir():
            raise SystemExit(f"missing upstream checkout: {name}")

    print("OK: split counts, disjoint WebShop IDs, and upstream checkouts verified")
    print(f"ALFWorld: {len(evo['tasks'])} evolution / {len(test['tasks'])} held-out")
    print(f"WebShop: {len(web_evo['tasks'])} evolution / {len(web_test['tasks'])} held-out")


if __name__ == "__main__":
    main()
