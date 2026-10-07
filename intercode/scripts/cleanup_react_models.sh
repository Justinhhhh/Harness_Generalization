#!/usr/bin/env bash
set -u

root=/playpen-shared/mohanz/Harness_Generalization/intercode
log="$root/react-results/cleanup.log"

count_tasks() {
    local result_root=$1
    python - "$result_root" <<'PY'
import glob, json, sys
total = 0
for path in glob.glob(sys.argv[1] + "/fs*/bash_react_10_turns.json"):
    try:
        total += len(json.load(open(path)))
    except Exception:
        pass
print(total)
PY
}

released4=0
released8=0
while [[ $released4 -eq 0 || $released8 -eq 0 ]]; do
    q4=$(count_tasks "$root/react-results/qwen3-4b-blackwell-220092-v2")
    q8=$(count_tasks "$root/react-results/qwen3-8b-blackwell-220301")
    printf '%s q4=%s/200 q8=%s/200\n' "$(date -u +'%FT%TZ')" "$q4" "$q8" >> "$log"

    if [[ $released4 -eq 0 && $q4 -eq 200 ]]; then
        scancel 220086 220092 220163 >> "$log" 2>&1 || true
        released4=1
        printf '%s released qwen3-4b jobs\n' "$(date -u +'%FT%TZ')" >> "$log"
    fi
    if [[ $released8 -eq 0 && $q8 -eq 200 ]]; then
        scancel 220300 220301 220302 >> "$log" 2>&1 || true
        released8=1
        printf '%s released qwen3-8b jobs\n' "$(date -u +'%FT%TZ')" >> "$log"
    fi
    sleep 60
done

mapfile -t containers < <(docker ps -a --format '{{.Names}}' | grep '^intercode-bash_ic_ctr')
if [[ ${#containers[@]} -gt 0 ]]; then
    timeout 60 docker rm -f "${containers[@]}" >> "$log" 2>&1 || true
fi
printf '%s react cleanup complete\n' "$(date -u +'%FT%TZ')" >> "$log"
