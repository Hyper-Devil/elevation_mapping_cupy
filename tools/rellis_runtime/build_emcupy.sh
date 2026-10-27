#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/activate_emcupy.sh"
cd "$EMCP_WORKSPACE"
catkin init
if [[ ! -d .catkin_tools/profiles/blackwell ]]; then
    catkin profile add blackwell
fi
catkin profile set blackwell
catkin config --extend /opt/ros/noetic --build-space build_blackwell --devel-space devel_blackwell \
    --cmake-args -DCMAKE_BUILD_TYPE=Release \
    "-DPYTHON_EXECUTABLE=$EMCP_PYTHON_PREFIX/bin/python" \
    "-DPYTHON_INCLUDE_DIR=$EMCP_PYTHON_PREFIX/include/python3.10" \
    "-DPYTHON_LIBRARY=$EMCP_PYTHON_PREFIX/lib/libpython3.10.so"
catkin build -j4 -p2 --no-status
