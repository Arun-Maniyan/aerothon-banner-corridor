#!/bin/bash
# TERMINAL 3 - camera bridge + banner search / alignment / corridor mission (with live HUD)
set -e
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash" 2>/dev/null || true
if ! pgrep -f "parameter_bridge.*camera/image_raw" > /dev/null; then
    echo "[SETUP] Starting ros_gz_bridge for /camera/image_raw ..."
    ros2 run ros_gz_bridge parameter_bridge '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image' &
    BRIDGE_PID=$!
    trap "kill $BRIDGE_PID 2>/dev/null || true" EXIT
    sleep 2
fi
echo "=========================================================="
echo " AeroTHON banner search -> align -> corridor mission"
echo "=========================================================="
python3 -u "$WS/scripts/banner_corridor_mission.py" "$@"
