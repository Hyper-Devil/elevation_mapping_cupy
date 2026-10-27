#!/usr/bin/env bash
# Source this file inside the container; all workspace helpers share this setup.
emcp_runtime_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EMCP_REPOSITORY="$(cd -- "$emcp_runtime_dir/../.." && pwd)"
export EMCP_WORKSPACE="${EMCP_WORKSPACE:-$(dirname -- "$(dirname -- "$EMCP_REPOSITORY")")}"
export EMCP_PYTHON_PREFIX="${EMCP_PYTHON_PREFIX:-/opt/emcupy-py310}"
source /opt/ros/noetic/setup.bash
if [[ -f "$EMCP_WORKSPACE/devel_blackwell/setup.bash" ]]; then
    source "$EMCP_WORKSPACE/devel_blackwell/setup.bash"
fi
export PATH="$EMCP_PYTHON_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$EMCP_PYTHON_PREFIX/lib:$EMCP_PYTHON_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export CUDA_PATH="$EMCP_PYTHON_PREFIX"
export PYTHONPATH="$EMCP_REPOSITORY/elevation_mapping_cupy/script:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
