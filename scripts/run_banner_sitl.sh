#!/bin/bash
# TERMINAL 2 - ArduPilot Copter SITL connected to Gazebo (JSON), outputs on 14550 / 14551
set -e
ARDUPILOT_DIR="${ARDUPILOT_DIR:-$HOME/ardupilot}"
cd "$ARDUPILOT_DIR"
EXTRA="--console --map"
[ "${NO_MAP:-0}" = "1" ] && EXTRA="--console"
echo "=========================================================="
echo " ArduPilot SITL (gazebo-iris, JSON) -> udp 14550 / 14551"
echo "=========================================================="
./Tools/autotest/sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON $EXTRA \
    --out=127.0.0.1:14550 --out=127.0.0.1:14551 "$@"
