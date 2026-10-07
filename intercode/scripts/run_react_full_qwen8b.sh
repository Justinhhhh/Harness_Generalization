#!/usr/bin/env bash
set -euo pipefail

root=/playpen-shared/mohanz/Harness_Generalization/intercode
python=/home/mohanz/Harness_Generalization/SWE-bench/.venv/bin/python
api_base=${REACT_API_BASE:-http://unites7.ib:30021/v1}
output_root="$root/react-results/qwen3-8b-blackwell-220301"

cd "$root"
mkdir -p "$output_root"
for fs in 1 2 3 4; do
    output="$output_root/fs${fs}"
    mkdir -p "$output"
    timeout 30 docker rm -f intercode-bash_ic_ctr intercode-bash_ic_ctr_eval >/dev/null 2>&1 || true
    OPENAI_API_KEY=EMPTY "$python" -m experiments.eval_react \
        --data_path "./data/nl2bash/nl2bash_fs_${fs}.json" \
        --env bash --image_name intercode-bash --log_dir "$output" \
        --max_turns 10 --api_base "$api_base" --model qwen3-8b \
        --setup_script "./docker/bash_scripts/setup_nl2b_fs_${fs}.sh"
done
