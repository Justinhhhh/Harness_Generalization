#!/usr/bin/env bash
# Build AgentBench OS images as linux/amd64 using a job-local rootless Docker.
#SBATCH --job-name=agentbench-os-amd64
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/playpen-shared/mohanz/Harness_Generalization/Life-Harness/AgentBench/logs/os-build-%j.out
#SBATCH --error=/playpen-shared/mohanz/Harness_Generalization/Life-Harness/AgentBench/logs/os-build-%j.err
set -euo pipefail

ROOT=/playpen-shared/mohanz/Harness_Generalization/Life-Harness/AgentBench
BUNDLE=/playpen-shared/mohanz/software/docker-rootless-29.7.2/docker-rootless-extras
SOCKET=/tmp/agentbench-docker-$SLURM_JOB_ID.sock
DATA=/tmp/agentbench-docker-data-$SLURM_JOB_ID
EXEC=/tmp/agentbench-docker-exec-$SLURM_JOB_ID
XDG=/tmp/agentbench-xdg-$SLURM_JOB_ID
OUT=/playpen-shared/mohanz/Harness_Generalization/agentbench-os-images-amd64.tar.gz
mkdir -p "$ROOT/logs" "$XDG" "$EXEC/rootlesskit"
export PATH="$BUNDLE:$PATH" XDG_RUNTIME_DIR="$XDG"
export DOCKER_HOST="unix://$SOCKET"
export DOCKERD_ROOTLESS_ROOTLESSKIT_NET=slirp4netns
export DOCKERD_ROOTLESS_ROOTLESSKIT_STATE_DIR="$EXEC/rootlesskit"
cleanup() { kill "${DOCKER_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT

dockerd-rootless.sh --host="unix://$SOCKET" --data-root="$DATA" \
  --exec-root="$EXEC" --storage-driver=vfs > "$ROOT/logs/dockerd-$SLURM_JOB_ID.log" 2>&1 &
DOCKER_PID=$!
for i in $(seq 1 90); do
  docker info >/dev/null 2>&1 && break
  kill -0 "$DOCKER_PID" 2>/dev/null || { cat "$ROOT/logs/dockerd-$SLURM_JOB_ID.log"; exit 1; }
  sleep 2
done
docker info >/dev/null

cd "$ROOT"
for name in default packages ubuntu; do
  docker build --network=host --platform linux/amd64 \
    -f "data/os_interaction/res/dockerfiles/$name" \
    data/os_interaction/res/dockerfiles --tag "local-os/$name"
done
docker image inspect local-os/default local-os/packages local-os/ubuntu \
  --format '{{.RepoTags}} {{.Architecture}}'
docker save local-os/default:latest local-os/packages:latest local-os/ubuntu:latest \
  | gzip -c > "$OUT"
gzip -t "$OUT"
echo "WROTE $OUT"
