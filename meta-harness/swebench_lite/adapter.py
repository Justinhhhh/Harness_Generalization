"""SWE-bench Lite domain adapter for the core Meta-Harness loop."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVOLUTION = ROOT / "data" / "splits" / "swebench_lite_evolution.jsonl"
HELDOUT = ROOT / "data" / "splits" / "swebench_lite_eval.jsonl"

def instance_ids(path: Path) -> tuple[str, ...]:
    return tuple(json.loads(line)["instance_id"] for line in path.read_text().splitlines() if line.strip())

def evolution_ids() -> tuple[str, ...]:
    return instance_ids(EVOLUTION)

def heldout_ids() -> tuple[str, ...]:
    return instance_ids(HELDOUT)

if __name__ == "__main__":
    print({"evolution": len(evolution_ids()), "heldout": len(heldout_ids())})
