# AeroTHON 2026 — Banner search, banner alignment & corridor entry

Add-on for the `QR-visual-Aligner` repo (ArduPilot SITL + Gazebo Harmonic + ROS 2 Humble).
It reuses the repo's drone model, downward camera, camera bridge, MAVLink controller and
QR alignment flow, and uses **your banner detector** (`siyi_banner_live_rapid_qrcam.py`,
RapidOCR) unchanged as the vision front end.

> Note: the repo is built on **ArduPilot**, not PX4 (ArduPilotPlugin in the drone model,
> `sim_vehicle.py`, GUIDED/BRAKE modes). Everything here follows the repo.

## Mission

| Stage | What happens |
|---|---|
| 1 Takeoff | Arm in GUIDED at (0, 0), climb to **5 m**, lock heading (North) |
| 2 Search | **Sway left** (body −Y = West) at 0.5 m/s, max **5 m**, detector running live |
| 3 Align | On first detection: **BRAKE** → project banner centre to NED → GUIDED fly-to → closed-loop visual centring (gain 0.75) + heading alignment → 3 s hold (repo's QR aligner, adapted) |
| 4 Descend | Down to **3 m** over the banner, re-centre once |
| 5 Surge | **10 m** straight ahead through the corridor at 3 m |
| 6 End | Hover (Ctrl+C or `q` lands) — or land with `--land-after` |

Live HUD of the bottom camera throughout: banner outline, **4 corners with pixel
coordinates (TL/TR/BR/BL)**, **centre C(x,y)**, line from crosshair to centre, metric
offset, mission state and telemetry. Green outline = OCR-confirmed, orange = tracker.

## World `aerothon_banner_corridor.sdf` (Gazebo X = East, Y = North)

| Object | Position / size |
|---|---|
| Drone spawn | (0, 0), facing North |
| Banner #00B140, 11.5 ft × 1 ft, "AeroTHON 2026" | centre E −7.4, N +2.0 (**2 m ahead**) |
| Corridor (walls + roof) | inner width 11.5 ft (3.5052 m), length 10 m, height 3.5 m, N 2.75 → 12.75, centred on the banner |
| Ground | grey (not green, so only the banner trips the green mask) |

Banner placement was calibrated with your detector on simulated 640×480 / 90° frames at
5 m, and the whole mission was flown against ArduPilot Copter SITL with a camera rendered
from the live SITL pose: first detection at **3.09 m and 3.17 m of sway** in two runs,
banner-centre estimate within 5 cm, centring lock at 4–11 px, corridor exit within 9 cm of
the centreline. (Gazebo's renderer differs slightly from that test, so expect ~3.0 ± 0.3 m.)
The first OCR hit happens before the whole banner is in frame (the detector accepts the text
while the left end of the banner is still cut off by the image edge; the mission
reconstructs the true centre from the visible end + the known 11.5 ft length).

Rule of thumb at 5 m: first-detection sway ≈ `(−banner_X − 4.45)` m. To shift it, change
the X of `aerothon_banner` **and** the three `corridor_*` models by the same amount.
Delete `corridor_roof` for an open-top corridor.

## Files (paths relative to the repo root)

```
src/aerothon_sim/models/aerothon_banner/          banner model + texture
src/aerothon_sim/worlds/aerothon_banner_corridor.sdf
scripts/siyi_banner_live_rapid_qrcam.py           your detector (unchanged, imported)
scripts/banner_sim_common.py                      ROS camera -> detector glue, corners, HUD
scripts/banner_corridor_mission.py                the mission
scripts/banner_cam_view.py                        camera + detector viewer only
scripts/test_banner_detection.py                  offline detector/geometry check
scripts/run_banner_sim.sh / run_banner_sitl.sh / run_banner_mission.sh / run_banner_cam_view.sh
```

## One-time setup (any WSL terminal)

```bash
# 1. put the add-on into your repo (the repo README calls it ~/aerothon_ws)
cd ~/aerothon_ws
unzip -o /mnt/c/Users/<YOUR_WINDOWS_USER>/Downloads/banner_addon.zip
chmod +x scripts/*.sh scripts/*.py

# 2. Python deps (system python3 that ROS Humble uses)
pip3 install --user rapidocr onnxruntime pymavlink
python3 -c "import numpy; print(numpy.__version__)"     # if 2.x and ROS errors appear:
# pip3 install --user "numpy<2"

# 3. rebuild the ROS package so the new model/world are installed too
source /opt/ros/humble/setup.bash
colcon build --symlink-install && source install/setup.bash

# 4. sanity checks
gz sdf -k src/aerothon_sim/worlds/aerothon_banner_corridor.sdf    # expect "Valid."
python3 scripts/test_banner_detection.py                         # expect FIRST DETECTION AT SWAY: ~2.9 m
```

Paths assumed by the launch scripts (override with env vars if yours differ):
`ARDUPILOT_DIR=$HOME/ardupilot`, `ARDUPILOT_GAZEBO=$HOME/gz_ws/src/ardupilot_gazebo`.

## Running — 3 terminals, in this order

**Terminal 1 — Gazebo**
```bash
cd ~/aerothon_ws && ./scripts/run_banner_sim.sh
```
Wait until the world shows the drone, the green banner and the corridor.

**Terminal 2 — ArduPilot SITL**
```bash
cd ~/aerothon_ws && ./scripts/run_banner_sitl.sh
```
Wait for `EKF3 IMU0 is using GPS` / `Pre-arm good` in the console (≈30–60 s).
(`NO_MAP=1 ./scripts/run_banner_sitl.sh` if the map window is slow in WSL.)

**Terminal 3 — Mission (+ live camera HUD)**
```bash
cd ~/aerothon_ws && ./scripts/run_banner_mission.sh
```
Useful options: `--land-after`, `--sway-speed 0.5`, `--max-sway 5`, `--alt 5`,
`--corridor-alt 3`, `--surge 10`, `--align-duration 3`, `--center-tol 12`, `--verbose`.

**Optional — viewer only** (no flying; check camera + detector, e.g. while you fly by
hand from MAVProxy with `mode guided`, `arm throttle`, `takeoff 5`):
```bash
cd ~/aerothon_ws && ./scripts/run_banner_cam_view.sh
```

Expected Terminal 3 output (abridged):
```
STAGE 1: TAKEOFF TO 5.0 m ... Reached 4.9x m
STAGE 2: SWAY LEFT (max 5.0 m) & SEARCH FOR BANNER
[BANNER DETECTED] text='AeroTHON 2026' after swaying ~3.0 m (partial view, centre reconstructed)
STAGE 3: BANNER ALIGNMENT ... [CENTERING] Lock acquired
STAGE 4: DESCEND TO 3.0 m
STAGE 5: SURGE 10.0 m THROUGH CORRIDOR ... Reached corridor end
MISSION COMPLETE
```

## Troubleshooting

* **HUD says WAITING FOR /camera/image_raw** — check `ros2 topic hz /camera/image_raw`
  in another terminal (after `source /opt/ros/humble/setup.bash`). Gazebo must be running.
* **Camera image black / Gazebo very slow in WSL** — try
  `export LIBGL_ALWAYS_SOFTWARE=1` before `run_banner_sim.sh` (slower but reliable),
  or update your Windows GPU driver for WSLg.
* **"Could not arm"** — SITL wasn't ready; wait for GPS lock in Terminal 2, the mission
  keeps retrying for 120 s and prints the `PreArm:` reason.
* **Detected too early/late** — move banner + corridor in X (rule of thumb above), or
  change `--sway-speed`.
* **Banner never detected but visible in HUD** — run with `--verbose` to see OCR groups.
  If the text renders upside-down in your Gazebo build it is still handled (180° retry).
* **Your detector and rotation** — it rejects the banner when it appears rotated by more
  than ~6–7° in the image (the green-blob fill-ratio filter in `green_candidate_boxes`).
  The mission holds heading so this never triggers here; keep it in mind on the real drone.
