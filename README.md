# Aerothon 2026 — Banner Corridor Mission & QR Visual Servoing

Autonomous drone simulation for **ArduPilot Copter** in **Gazebo Sim 8 (Harmonic)** with **ROS 2 Humble**. It contains two missions that share the same camera and vision stack:

1. **Banner corridor mission** — take off, sway sideways while an OCR detector looks for a green "AeroTHON 2026" banner, align over it, descend, and fly 10 m into a covered corridor. A live camera window shows the banner outline, labelled corners and centre.
2. **QR lawnmower mission** — survey a field, match a reference QR code by visual similarity (rotation-invariant, no text decoding), and hold a precise hover over the match.

> Based on [leeIITM/QR-visual-Aligner](https://github.com/leeIITM/QR-visual-Aligner), used with the author's permission. The QR mission, its alignment logic and the ROS package layout come from that project; the banner corridor mission, world and live camera feed were added on top.

---

## Contents

- [How it works](#how-it-works)
- [Repository structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation (one time)](#installation-one-time)
- [First-time SITL setup](#first-time-sitl-setup)
- [Running the banner corridor mission](#running-the-banner-corridor-mission)
- [Running the QR lawnmower mission](#running-the-qr-lawnmower-mission)
- [Command-line arguments](#command-line-arguments)
- [Configuration and tuning](#configuration-and-tuning)
- [Troubleshooting](#troubleshooting)
- [Credits](#credits)

---

## How it works

### Banner corridor mission

| Phase | What happens |
|---|---|
| Takeoff | Arms in GUIDED mode and climbs to 5 m. |
| Sway | Moves left (up to 5 m) while the OCR detector looks for the banner text. |
| Align | On first detection: BRAKE, estimate where the banner is, fly to it, then re-centre over it in small steps. A heading correction squares the nose to the banner. |
| Descend | Goes down to 3 m and re-centres once. |
| Surge | Flies 10 m into the corridor and hovers (or lands with `--land-after`). |

When the banner first appears it is cut off at the image edge. The mission works out the true centre from the visible end and the known banner length (11.5 ft), with about 3 cm error in testing.

**World layout**
- Grey ground, drone at (0, 0) facing north.
- Green (`#00B140`) banner, 11.5 ft × 1 ft, with the text "AeroTHON 2026". It sits 2 m ahead and 7.4 m to the left.
- Corridor 11.5 ft wide, 10 m long, 3.5 m tall walls and roof. Its mouth starts 0.6 m past the banner so the propellers clear the roof edge while descending to 3 m.

### QR lawnmower mission

- **Blue takeoff zone** (50 m × 10 m): the drone takes off to 4 m and memorises the reference QR.
- **Green search zone** (50 m × 40 m): serpentine survey, separated by a 5 m buffer to avoid false self-detections.
- **Rotation-invariant similarity:** the candidate QR quad is perspective-warped to 210 × 210 px, sampled as a 21 × 21 module matrix, and correlated at 0°, 90°, 180° and 270°. A match triggers at similarity ≥ 0.88.
- **Position-based visual servoing:** BRAKE, pinhole reprojection to a world position, then closed-loop centring to sub-centimetre alignment, followed by a timed hover.

---

## Repository structure

```
.
├── README.md
├── scripts/
│   ├── banner_corridor_mission.py        # Banner mission state machine
│   ├── banner_sim_common.py              # Shared helpers (MAVLink, camera, geometry)
│   ├── banner_cam_view.py                # Standalone camera + detector viewer
│   ├── siyi_banner_live_rapid_qrcam.py   # OCR banner detector
│   ├── test_banner_detection.py          # Offline detector test (no flight)
│   ├── run_banner_sim.sh                 # Launch Gazebo (banner world)
│   ├── run_banner_sitl.sh                # Launch ArduPilot SITL
│   ├── run_banner_mission.sh             # Launch banner mission + camera bridge
│   ├── run_banner_cam_view.sh            # Camera viewer only
│   ├── lawnmower_qr_mission.py           # QR mission state machine
│   ├── fly_to.py                         # NED waypoint utility
│   ├── run_sim.sh                        # Launch Gazebo (QR world)
│   ├── run_sitl.sh                       # Launch ArduPilot SITL (QR)
│   ├── run_mission.sh                    # Launch QR mission
│   └── run_cam_view.sh                   # QR camera viewer
└── src/
    └── aerothon_sim/                     # ROS 2 package
        ├── CMakeLists.txt
        ├── package.xml
        ├── models/
        │   ├── aerothon_banner/          # Banner model + texture
        │   └── ...                       # QR tile models
        └── worlds/
            ├── aerothon_banner_corridor.sdf   # Banner + corridor world
            └── ...                            # QR field world
```

---

## Prerequisites

- Ubuntu 22.04 (native or WSL2 with WSLg for GUI windows)
- ROS 2 Humble
- Gazebo Sim 8 (Harmonic) — check with `gz sim --version` (needs 8.x)
- A GPU driver that supports OpenGL if you want Gazebo to run smoothly

PX4 is **not** used. This project runs on ArduPilot.

---

## Installation (one time)

### 1. ArduPilot SITL

```bash
cd ~
sudo apt update && sudo apt install -y git unzip
git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git
cd ardupilot
Tools/environment_install/install-prereqs-ubuntu.sh -y
. ~/.profile

./waf configure --board sitl
./waf copter
```

> If `apt` reports it is waiting for a lock held by `unattended-upgr`, that is Ubuntu's background updater. Wait for it to finish, or stop it with `sudo systemctl stop unattended-upgrades` and re-run.

### 2. ArduPilot Gazebo plugin

```bash
sudo apt install -y libgz-sim8-dev rapidjson-dev libopencv-dev \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl

mkdir -p ~/gz_ws/src && cd ~/gz_ws/src
git clone https://github.com/ArduPilot/ardupilot_gazebo.git
cd ardupilot_gazebo
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j4
```

If the build runs out of memory, use `make -j2`.

### 3. ROS–Gazebo bridge

```bash
source /opt/ros/humble/setup.bash
ros2 pkg list | grep ros_gz
```

If nothing is printed:

```bash
sudo apt install -y ros-humble-ros-gzharmonic
```

### 4. This repository

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git aerothon_ws
cd ~/aerothon_ws
chmod +x scripts/*.sh scripts/*.py
```

The folder must be named `aerothon_ws` because the scripts and commands below assume `~/aerothon_ws`.

### 5. Python dependencies and build

```bash
pip3 install --user rapidocr onnxruntime pymavlink opencv-python numpy

cd ~/aerothon_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### 6. Check the detector (no flight)

```bash
python3 scripts/test_banner_detection.py
```

Expected output includes `FIRST DETECTION AT SWAY: ~2.9 m`.

---

## First-time SITL setup

The first time SITL runs, the copter has no frame configured and refuses to arm with `Arm: Motors: Check frame class and type`. Fix it once:

1. Start Gazebo and SITL (see below).
2. At the `STABILIZE>` prompt in the SITL terminal:
   ```
   param set FRAME_CLASS 1
   param set FRAME_TYPE 1
   param save
   ```
3. Press Ctrl+C in the SITL terminal and start SITL again. (`reboot` often kills SITL without restarting it, so a manual restart is more reliable.)
4. Confirm with `param show FRAME_CLASS` and `param show FRAME_TYPE` (both should be 1), then test:
   ```
   mode guided
   arm throttle
   ```

The values are stored in `eeprom.dat` in the folder SITL starts from. Do not delete that file, or repeat this step.

---

## Running the banner corridor mission

Open **three** terminals. Start them in this order and wait for each "ready" sign.

| Terminal | Command | Wait for |
|---|---|---|
| 1 | `cd ~/aerothon_ws && source /opt/ros/humble/setup.bash && source install/setup.bash && ./scripts/run_banner_sim.sh` | Gazebo window with drone, banner and corridor |
| 2 | `cd ~/aerothon_ws && ./scripts/run_banner_sitl.sh` | `EKF3 IMU0 is using GPS` |
| 3 | `cd ~/aerothon_ws && source /opt/ros/humble/setup.bash && source install/setup.bash && ./scripts/run_banner_mission.sh` | Live camera window opens, drone takes off |

Add `--land-after` to terminal 3 to land at the end of the corridor. Otherwise the drone hovers there until you press Ctrl+C.

To view only the camera and detector without flying:

```bash
./scripts/run_banner_cam_view.sh
```

**Stopping:** press Ctrl+C in terminals 3, 2, then 1. If processes linger:

```bash
pkill -f arducopter; pkill -f mavproxy; pkill -f sim_vehicle; pkill -f "gz sim"; pkill -f ruby; pkill -f ros_gz_bridge
```

---

## Running the QR lawnmower mission

```bash
# Terminal 1
cd ~/aerothon_ws && ./scripts/run_sim.sh

# Terminal 2
cd ~/aerothon_ws && ./scripts/run_sitl.sh

# Terminal 3
cd ~/aerothon_ws && ./scripts/run_mission.sh --align-duration 15.0 --speed 2.0
```

### QR mission arguments

| Argument | Type | Default | Description |
|:---|:---|:---|:---|
| `--alt` | float | `4.0` | Survey altitude in metres AGL. |
| `--align-duration` | float | `10.0` | Seconds to hold the aligned hover. |
| `--speed` | float | `1.5` | Lawnmower traversal speed in m/s. |
| `--step` | float | `5.0` | East spacing between serpentine sweeps in metres. |
| `--match-thresh` | float | `0.88` | Minimum similarity [0.0 – 1.0]. |
| `--center-tol` | float | `25.0` | Centring tolerance in pixels before declaring lock. |
| `--land-after` | flag | off | Land automatically after alignment. |
| `--no-camera` | flag | off | Disable the live OpenCV window. |

### Banner mission arguments

| Argument | Description |
|:---|:---|
| `--land-after` | Land at the end of the corridor instead of hovering. |

Run `./scripts/run_banner_mission.sh --help` to see any further options.

---

## Configuration and tuning

- **Custom install paths:** the scripts expect ArduPilot at `~/ardupilot` and the plugin at `~/gz_ws/src/ardupilot_gazebo`. If yours are elsewhere, set these before running:
  ```bash
  export ARDUPILOT_DIR=/path/to/ardupilot
  export ARDUPILOT_GAZEBO=/path/to/ardupilot_gazebo
  ```
- **Detection point:** if the banner is detected noticeably earlier or later than about 3 m of sway, move the banner and the three corridor pieces sideways by the same amount in `src/aerothon_sim/worlds/aerothon_banner_corridor.sdf`. At 5 m altitude, detection sway ≈ (−banner X − 4.45) m.
- **Rebuilding:** scripts are symlinked, so edits take effect immediately. Re-run `colcon build --symlink-install` only after changing the ROS package or world files.
- **Rotation limit:** the OCR detector misses the banner once it appears rotated by more than about 7° in the image. The mission holds heading so this does not happen in simulation, but it matters on a real drone.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Arm: Motors: Check frame class and type` | Set `FRAME_CLASS 1` and `FRAME_TYPE 1`; see [First-time SITL setup](#first-time-sitl-setup). |
| `reboot` leaves `Connection refused` | SITL died. Press Ctrl+C and run `./scripts/run_banner_sitl.sh` again. Keep `eeprom.dat`. |
| `time moved backwards. Restarting timer` | Harmless WSL clock warning. Ignore it. |
| `PreArm:` messages | Wait 30–60 s after `EKF3 IMU0 is using GPS`, then retry. |
| No Gazebo or camera window on WSL | Needs WSLg (Windows 11 or recent Windows 10). Test with `gz sim`. |
| Gazebo very slow on WSL | Update the Windows GPU driver and run `wsl --update`. |
| `apt` stuck on `unattended-upgr` lock | Wait, or run `sudo systemctl stop unattended-upgrades`. |
| `ros_gz` packages missing | `sudo apt install -y ros-humble-ros-gzharmonic` |
| Detection earlier or later than ~3 m | See [Configuration and tuning](#configuration-and-tuning). |

---

## Credits

- **QR-visual-Aligner** by [leeIITM](https://github.com/leeIITM/QR-visual-Aligner) — QR mission, visual servoing, camera stack and package layout. Used with permission.
- [ArduPilot](https://ardupilot.org/), [ardupilot_gazebo](https://github.com/ArduPilot/ardupilot_gazebo), [Gazebo Sim](https://gazebosim.org/) and [ROS 2](https://docs.ros.org/en/humble/) are used under their own licences.

## Licence

Add a licence once you have agreed one with the original author. Until then, all rights remain with the respective authors.
