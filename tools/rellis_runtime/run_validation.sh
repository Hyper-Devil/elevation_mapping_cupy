#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/activate_emcupy.sh"
export EMCP_FIXTURES="${1:-$EMCP_WORKSPACE/validation/fixtures}"
export EMCP_VALIDATION_OUTPUT="${2:-$EMCP_WORKSPACE/validation/runs/$(date -u +%Y%m%dT%H%M%S)}"
[[ -f "$EMCP_FIXTURES/metadata.json" && -f "$EMCP_FIXTURES/rgb.npy" ]]
mkdir -p "$(dirname -- "$EMCP_VALIDATION_OUTPUT")"
mkdir "$EMCP_VALIDATION_OUTPUT"
python -m pytest "$emcp_runtime_dir/test_runtime.py" \
    "$EMCP_REPOSITORY/elevation_mapping_cupy/tests/test_builder_plugins.py" \
    -v --tb=short --junitxml="$EMCP_VALIDATION_OUTPUT/runtime_tests.xml"
bash "$emcp_runtime_dir/run_ros_smoke.sh"
