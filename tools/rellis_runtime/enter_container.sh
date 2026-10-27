#!/usr/bin/env bash
set -euo pipefail
exec docker exec -it -w /home/root/catkin_emcp "${EMCP_CONTAINER:-emcupy}" \
    bash --rcfile /home/root/catkin_emcp/src/elevation_mapping_cupy/tools/rellis_runtime/activate_emcupy.sh -i
