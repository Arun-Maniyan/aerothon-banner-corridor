#!/bin/bash
# OPTIONAL - live bottom camera + banner detector only (no flight control)
set -e
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
if ! pgrep -f "parameter_bridge.*camera/image_raw" > /dev/null; then
    ros2 run ros_gz_bridge parameter_bridge '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image' &
    BRIDGE_PID=$!
    trap "kill $BRIDGE_PID 2>/dev/null || true" EXIT
    sleep 2
fi
python3 -u "$WS/scripts/banner_cam_view.py" "$@"
