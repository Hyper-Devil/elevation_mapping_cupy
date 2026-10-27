#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/activate_emcupy.sh"
: "${EMCP_FIXTURES:?run through run_validation.sh}"
: "${EMCP_VALIDATION_OUTPUT:?run through run_validation.sh}"
export EMCP_ROS_PORT="${EMCP_ROS_PORT:-11641}"
export ROS_MASTER_URI="http://127.0.0.1:$EMCP_ROS_PORT"
export ROS_IP=127.0.0.1
export ROS_HOSTNAME=localhost
export ROS_LOG_DIR="$EMCP_VALIDATION_OUTPUT/ros_logs"
/usr/bin/python3 - <<'PY'
import os, socket
with socket.socket() as sock:
    sock.bind(('127.0.0.1', int(os.environ['EMCP_ROS_PORT'])))
PY
master_pid=
launch_pid=
cleanup() {
    for pid in "$launch_pid" "$master_pid"; do
        [[ -n "$pid" ]] || continue
        kill -INT -- "-$pid" 2>/dev/null || true
        for _ in {1..50}; do
            kill -0 -- "-$pid" 2>/dev/null || break
            sleep .1
        done
        kill -TERM -- "-$pid" 2>/dev/null || true
        sleep .2
        kill -KILL -- "-$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT
setsid roscore -p "$EMCP_ROS_PORT" > "$EMCP_VALIDATION_OUTPUT/ros_master.log" 2>&1 &
master_pid=$!
/usr/bin/python3 - <<'PY'
import os, time, xmlrpc.client
for _ in range(100):
    try:
        assert xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI']).getPid('/emcupy_smoke_check')[0] == 1
        break
    except Exception:
        time.sleep(.1)
else:
    raise RuntimeError('isolated ROS master failed')
PY
setsid roslaunch "$emcp_runtime_dir/ros_smoke.launch" > "$EMCP_VALIDATION_OUTPUT/ros_node.log" 2>&1 &
launch_pid=$!
timeout 180 /usr/bin/python3 "$emcp_runtime_dir/ros_smoke.py" > "$EMCP_VALIDATION_OUTPUT/ros_smoke.log" 2>&1
