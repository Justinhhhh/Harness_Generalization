"""Shared Meta-Harness outer-loop primitives, aligned with TB2 reference."""
from __future__ import annotations
import ast, json
from pathlib import Path

def validate_harness(path: Path) -> tuple[bool,str]:
    try:
        tree=ast.parse(path.read_text()); cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Harness')
        return True,'ok'
    except Exception as e: return False,str(e)

def update_frontier(path: Path, name: str, score: float, metadata: dict | None = None) -> dict:
    state=json.loads(path.read_text()) if path.exists() else {'_best':{'agent':'baseline','pass_rate':0.0}}
    best=state.get('_best',{}).get('pass_rate',0.0); entry={'pass_rate':score,**(metadata or {})}
    state.setdefault('candidates',{})[name]=entry
    if score>best: state['_best']={'agent':name,'pass_rate':score}
    path.write_text(json.dumps(state,indent=2)+'\n'); return state

def append_summary(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f: f.write(json.dumps(row,sort_keys=True)+'\n')
