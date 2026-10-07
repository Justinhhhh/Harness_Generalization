"""Materialize the canonical ALFWorld evolution and held-out manifests."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT.parent.parent / "data" / "splits"
OUT = ROOT / "data" / "alfworld"

for source, target in (("alfworld_evolution.json", "alfworld_evolution.json"),
                       ("alfworld_test.json", "alfworld_heldout.json")):
    raw = json.loads((CANONICAL / source).read_text())
    grouped = defaultdict(list)
    for task in raw["tasks"]:
        grouped[task["task_type"]].append(task["metadata"]["gamefile"])
    (OUT / target).write_text(json.dumps(dict(grouped), indent=2) + "\n")
    print(target, sum(map(len, grouped.values())))
