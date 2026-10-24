#!/bin/bash
# TERMINAL 1 - Gazebo Harmonic with the banner + corridor world
set -e
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
ARDUPILOT_GAZEBO="${ARDUPILOT_GAZEBO:-$HOME/gz_ws/src/ardupilot_gazebo}"
if [ ! -d "$ARDUPILOT_GAZEBO/build" ]; then
  echo "[ERROR] ardupilot_gazebo plugin not found at $ARDUPILOT_GAZEBO/build"
  echo "        export ARDUPILOT_GAZEBO=/path/to/ardupilot_gazebo and re-run."
  exit 1
fi
export GZ_SIM_SYSTEM_PLUGIN_PATH="$ARDUPILOT_GAZEBO/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}"
export GZ_SIM_RESOURCE_PATH="$WS/src/aerothon_sim/models:$WS/src/aerothon_sim/worlds:$ARDUPILOT_GAZEBO/models:$ARDUPILOT_GAZEBO/worlds:${GZ_SIM_RESOURCE_PATH}"
echo "=========================================================="
echo " Gazebo: AeroTHON banner + corridor world"
echo "=========================================================="
gz sim -v4 -r aerothon_banner_corridor.sdf
