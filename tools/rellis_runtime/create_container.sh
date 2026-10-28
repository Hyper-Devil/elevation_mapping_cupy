#!/usr/bin/env bash
set -euo pipefail
runtime_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace="${1:-$(cd -- "$runtime_dir/../../../.." && pwd)}"
workspace="$(realpath -- "$workspace")"
container="${EMCP_CONTAINER:-emcupy}"
runtime_image="${EMCP_IMAGE:-emcupy:blackwell-20261026}"
expected="${EMCP_IMAGE_ID:-sha256:e273af8a0756bab4dcd385739420d3b110f3c95e0a2b0e437675e46ba755cfad}"
[[ -f "$workspace/src/elevation_mapping_cupy/tools/rellis_runtime/activate_emcupy.sh" ]]
if docker container inspect "$container" >/dev/null 2>&1; then
    existing_image="$(docker inspect "$container" --format '{{.Image}}')"
    existing_workspace="$(docker inspect "$container" --format '{{range .Mounts}}{{if eq .Destination "/home/root/catkin_emcp"}}{{.Source}}{{end}}{{end}}')"
    if [[ "$existing_image" != "$expected" || "$existing_workspace" != "$workspace" ]]; then
        echo "Container $container has a different image or workspace; inspect its configuration before proceeding." >&2
        exit 1
    fi
    gpu_requests="$(docker inspect "$container" --format '{{json .HostConfig.DeviceRequests}}')"
    if [[ "$gpu_requests" != *'"gpu"'* ]]; then
        echo "Container $container does not expose a GPU." >&2
        exit 1
    fi
    state="$(docker inspect "$container" --format '{{.State.Status}}')"
    case "$state" in
        running) ;;
        exited|created) docker start "$container" >/dev/null ;;
        *) echo "Container $container is $state; inspect it before reuse." >&2; exit 1 ;;
    esac
    echo "Reusing container $container with workspace $workspace."
    exit 0
fi
actual="$(docker image inspect "$runtime_image" --format '{{.Id}}')"
if [[ "$actual" != "$expected" ]]; then
    echo "Image ID mismatch: expected $expected, got $actual" >&2
    exit 1
fi
docker run -d --name "$container" --init --gpus all --network host \
    --mount "type=bind,src=$workspace,dst=/home/root/catkin_emcp" \
    --workdir /home/root/catkin_emcp "$runtime_image" sleep infinity
