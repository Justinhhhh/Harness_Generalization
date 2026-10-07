#!/bin/bash
set -u

root=/playpen-shared/mohanz/Harness_Generalization/intercode
q4="$root/batch-results/qwen3-4b-ada-full-20261001"
q8="$root/batch-results/qwen3-8b-ada-full-20261001"
log="$root/batch-results/cleanup-when-complete.log"

while true; do
    q4_traj=$(find "$q4" -maxdepth 1 -name 'task-*.json' ! -name '*.reward.json' | wc -l)
    q4_reward=$(find "$q4" -maxdepth 1 -name '*.reward.json' | wc -l)
    q8_traj=$(find "$q8" -maxdepth 1 -name 'task-*.json' ! -name '*.reward.json' | wc -l)
    q8_reward=$(find "$q8" -maxdepth 1 -name '*.reward.json' | wc -l)
    printf '%s q4=%s/%s q8=%s/%s\n' "$(date -u +'%FT%TZ')" \
        "$q4_traj" "$q4_reward" "$q8_traj" "$q8_reward" >> "$log"

    if [[ "$q4_traj" -eq 200 && "$q4_reward" -eq 200 && -f "$q4/summary.json" \
       && "$q8_traj" -eq 200 && "$q8_reward" -eq 200 && -f "$q8/summary.json" ]]; then
        break
    fi
    sleep 60
done

mapfile -t containers < <(docker ps -a --format '{{.Names}}' | \
    grep -E '^intercode-bash-(qwen3-(4b|8b)|replay-qwen3-(4b|8b))_ic_ctr')
if [[ ${#containers[@]} -gt 0 ]]; then
    timeout 60 docker rm -f "${containers[@]}" >> "$log" 2>&1 || true
fi

scancel 216923 216924 >> "$log" 2>&1 || true
printf '%s cleanup complete; results retained\n' "$(date -u +'%FT%TZ')" >> "$log"
