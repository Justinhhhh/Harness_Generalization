import json
from pathlib import Path

src=Path('/home/mohanz/Harness_Generalization/data/splits/swebench_lite_evolution.jsonl')
out=Path(__file__).parent/'runs'/'smoke-selection.jsonl'
out.parent.mkdir(parents=True,exist_ok=True)
rows=[json.loads(x) for x in src.read_text().splitlines() if x.strip()][:10]
out.write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
print({'selected':len(rows),'output':str(out),'ids':[x['instance_id'] for x in rows]})
