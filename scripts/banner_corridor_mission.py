#!/usr/bin/env python3
"""
AeroTHON 2026 - Banner search, banner alignment and corridor entry (ArduPilot SITL + Gazebo)
--------------------------------------------------------------------------------------------
STAGE 1  Takeoff to --alt (5 m) at the start point.
STAGE 2  SEARCH: sway LEFT (body frame) at --sway-speed, at most --max-sway (5 m), while
         YOUR detector (siyi_banner_live_rapid_qrcam.py, RapidOCR) looks for the green
         "AeroTHON 2026" banner in the downward camera.
STAGE 3  BANNER ALIGNMENT (the repo's QR aligner, adapted to the banner):
           BRAKE -> pinhole projection of the banner centre to NED -> GUIDED fly-to
           -> closed-loop visual centring (gain 0.75) + heading alignment (nose
           perpendicular to the banner's long side) -> timed hold.
STAGE 4  DESCEND to --corridor-alt (3 m) over the banner (and re-centre once).
STAGE 5  SURGE straight ahead --surge metres (10 m = corridor length).
STAGE 6  Hold (or land with --land-after).

A live HUD of the bottom camera runs the whole time and shows the banner outline,
its four corners (pixel coordinates) and its centre.
"""
import argparse
import math
import os
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from pymavlink import mavutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from lawnmower_qr_mission import DroneController          # noqa: E402  (repo's MAVLink controller)
import banner_sim_common as bsc                            # noqa: E402

WINDOW = "Drone Downward Camera (Banner Mission HUD)"
YAW_MASK = 0b100111111000      # use position + yaw; ignore velocity, accel, yaw-rate


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ======================================================================================
# Drone controller: the repo's DroneController + yaw-locked setpoints, robust arming,
# STATUSTEXT printing and mode tracking.
# ======================================================================================
class BannerDrone(DroneController):
    def __init__(self, url=None):
        super().__init__(url)
        self.mode = ""
        self._last_text = ""
        self._request_streams()

    def _request_streams(self):
        for msg_id in (mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED,
                       mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE):
            self.m.mav.command_long_send(self.sys, self.comp,
                                         mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                                         0, msg_id, 50000, 0, 0, 0, 0, 0)   # 20 Hz

    def poll(self):
        now = time.time()
        if now - self._last_hb > 1.0:
            self.m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                      mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            self._last_hb = now
        while True:
            msg = self.m.recv_match(blocking=False)
            if msg is None:
                break
            t = msg.get_type()
            if t == "LOCAL_POSITION_NED":
                self.ned = [msg.x, msg.y, msg.z]
            elif t == "ATTITUDE":
                self.yaw = msg.yaw
            elif t == "HEARTBEAT":
                if msg.type == mavutil.mavlink.MAV_TYPE_GCS or msg.get_srcSystem() != self.sys:
                    continue                      # ignore MAVProxy / other GCS heartbeats
                self.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                self.mode = mavutil.mode_string_v10(msg)
            elif t == "STATUSTEXT":
                txt = msg.text.strip() if isinstance(msg.text, str) else str(msg.text)
                if txt and txt != self._last_text:
                    self._last_text = txt
                    print(f"\n[AUTOPILOT] {txt}")

    def setpoint_yaw(self, north, east, down, yaw):
        self.m.mav.set_position_target_local_ned_send(
            0, self.sys, self.comp, mavutil.mavlink.MAV_FRAME_LOCAL_NED, YAW_MASK,
            north, east, down, 0, 0, 0, 0, 0, 0, yaw, 0)

    def arm_and_takeoff_robust(self, alt, timeout=120.0, idle=None):
        """Keeps retrying GUIDED + ARM until the EKF/GPS pre-arm checks pass."""
        idle = idle or (lambda: None)
        t0, last_try = time.time(), 0.0
        print("[DRONE] Waiting for pre-arm checks (EKF/GPS) and arming in GUIDED...")
        while not self.armed:
            if time.time() - t0 > timeout:
                raise RuntimeError("Could not arm within %.0fs (see [AUTOPILOT] messages)" % timeout)
            if time.time() - last_try > 2.0:
                last_try = time.time()
                self.set_mode("GUIDED")
                self.m.mav.command_long_send(self.sys, self.comp,
                                             mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                                             0, 1, 0, 0, 0, 0, 0, 0)
            self.poll()
            idle()
            time.sleep(0.05)
        print("[DRONE] Motors ARMED!")
        time.sleep(0.5)
        print(f"[DRONE] Taking off to {alt:.1f} m ...")
        for _ in range(3):
            self.m.mav.command_long_send(self.sys, self.comp, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                                         0, 0, 0, 0, 0, 0, 0, alt)
            time.sleep(0.1)


# ======================================================================================
# Mission
# ======================================================================================
class BannerCorridorMission:
    def __init__(self, args):
        self.args = args
        self.node = rclpy.create_node("aerothon_banner_corridor_mission")
        self.source = bsc.RosFrameSource(self.node, args.image_topic)
        self._spin = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self._spin.start()

        print("[VISION] Loading YOUR banner detector (RapidOCR) ...")
        self.pipe = bsc.start_banner_pipeline(self.source, args.ocr_threads, args.verbose)

        self.drone = None
        self.state_text = "INITIALIZING"
        self.countdown = 0.0
        self.home = (0.0, 0.0)
        self.yaw_lock = 0.0
        self.sway_done = 0.0
        self._meas_cache = (None, None)          # (meas, (frame_id, snapshot key))
        if args.show_camera:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, 960, 720)

    # ------------------------------------------------------------------ helpers
    @property
    def alt(self):
        return -self.drone.ned[2] if self.drone else 0.0

    def banner(self):
        """Current banner measurement (centre, 4 corners, angle) or None."""
        frame, fid = self.source.latest()
        snap = self.pipe.state.snapshot()
        key = (fid, None if snap is None else (snap["box"], snap["source"]))
        if self._meas_cache[1] == key:
            return self._meas_cache[0], frame
        meas = bsc.measure_banner(frame, snap, self.alt) if snap else None
        self._meas_cache = (meas, key)
        return meas, frame

    def tick(self, sp=None):
        """One control-loop tick: telemetry, optional setpoint, HUD."""
        self.drone.poll()
        if sp is not None:
            self.drone.setpoint_yaw(sp[0], sp[1], sp[2], self.yaw_lock)
        meas, frame = self.banner()
        if self.args.show_camera:
            n, e = self.drone.ned[0], self.drone.ned[1]
            tele = (f"Alt {self.alt:4.2f}m  N {n:+5.2f} E {e:+5.2f}  "
                    f"yaw {math.degrees(self.drone.yaw):+5.1f}  {self.drone.mode}")
            img = bsc.draw_hud(frame, meas, self.state_text, tele, self.countdown,
                               self.source.connected, self.alt)
            cv2.imshow(WINDOW, img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt
        return meas

    def run_for(self, seconds, sp=None, dt=0.05):
        t0 = time.time()
        while time.time() - t0 < seconds:
            self.tick(sp)
            time.sleep(dt)

    def banner_to_ned(self, meas):
        fw, fh = meas["size"]
        dn, de = bsc.pixel_to_ned_offset(meas["u"], meas["v"], fw, fh, self.alt, self.drone.yaw)
        return self.drone.ned[0] + dn, self.drone.ned[1] + de

    def fly_to(self, target, timeout=15.0, tol=0.25, speed=None):
        """Slewed setpoint (as in the repo) towards target=(N, E, D)."""
        speed = speed or self.args.speed
        sp = [self.drone.ned[0], self.drone.ned[1], self.drone.ned[2]]
        t0, dt = time.time(), 0.05
        while time.time() - t0 < timeout:
            d_sp = math.dist(target, sp)
            step = min(speed * dt, d_sp)
            if d_sp > 1e-4:
                sp = [s + (t - s) * step / d_sp for s, t in zip(sp, target)]
            self.tick(sp)
            if d_sp < 0.05 and math.dist(target, self.drone.ned) < tol:
                return True
            time.sleep(dt)
        return False

    def wait_fresh_banner(self, timeout=2.5, sp=None, need_full=False):
        t0 = time.time()
        while time.time() - t0 < timeout:
            meas = self.tick(sp)
            if meas is not None and not (need_full and meas["partial"]):
                return meas
            time.sleep(0.05)
        return None

    # ------------------------------------------------------------- alignment
    def center_over_banner(self, target, iterations, tol_px, label):
        """Repo's closed-loop visual servoing, applied to the banner centre + heading."""
        self.state_text = f"BANNER ALIGN: {label}"
        for it in range(1, iterations + 1):
            meas = self.wait_fresh_banner(sp=target)
            if meas is None:
                print(f"[CENTERING] Iter {it}/{iterations}: banner not visible, holding...")
                continue
            fw, fh = meas["size"]
            err_px = math.hypot(meas["u"] - fw / 2.0, meas["v"] - fh / 2.0)
            yaw_err = 0.0 if meas["partial"] else meas["angle"]
            print(f"[CENTERING] Iter {it}/{iterations}: centre ({meas['u']:.1f},{meas['v']:.1f})px "
                  f"| offset {err_px:.1f}px | banner angle {math.degrees(yaw_err):+.1f} deg "
                  f"| {'PARTIAL' if meas['partial'] else 'full'} view")
            if err_px <= tol_px and abs(yaw_err) <= math.radians(self.args.yaw_tol) and not meas["partial"]:
                print(f"[CENTERING] Lock acquired ({err_px:.1f}px <= {tol_px:.0f}px).")
                return target, True
            bn, be = self.banner_to_ned(meas)
            target[0] += (bn - self.drone.ned[0]) * 0.75
            target[1] += (be - self.drone.ned[1]) * 0.75
            if abs(yaw_err) > math.radians(self.args.yaw_tol):
                self.yaw_lock = wrap_pi(self.drone.yaw + yaw_err)
                print(f"[CENTERING] Heading correction -> {math.degrees(self.yaw_lock):+.1f} deg")
            self.run_for(self.args.settle, target)
        return target, False

    # ------------------------------------------------------------------ main
    def run(self):
        a = self.args
        self.state_text = "CONNECTING"
        self.drone = BannerDrone(a.url)

        # STAGE 1 -------------------------------------------------------------
        print(f"\n{'=' * 60}\nSTAGE 1: TAKEOFF TO {a.alt:.1f} m\n{'=' * 60}")
        self.state_text = "ARMING"
        self.drone.arm_and_takeoff_robust(a.alt, idle=lambda: self.tick())
        self.state_text = f"TAKEOFF ({a.alt:.1f}m)"
        t0 = time.time()
        while time.time() - t0 < 40.0:
            self.tick()
            print(f"[STATUS] Altitude: {self.alt:.2f} / {a.alt:.1f} m", end="\r")
            if self.alt >= a.alt * 0.95:
                break
            time.sleep(0.05)
        print(f"\n[STATUS] Reached {self.alt:.2f} m")
        self.home = (self.drone.ned[0], self.drone.ned[1])
        self.yaw_lock = self.drone.yaw
        start = [self.home[0], self.home[1], -a.alt]
        self.run_for(2.0, start)
        print(f"[STATUS] Start point N={start[0]:+.2f} E={start[1]:+.2f}, heading locked "
              f"at {math.degrees(self.yaw_lock):+.1f} deg")

        # STAGE 2 -------------------------------------------------------------
        print(f"\n{'=' * 60}\nSTAGE 2: SWAY LEFT (max {a.max_sway:.1f} m) & SEARCH FOR BANNER\n{'=' * 60}")
        self.state_text = "SEARCH: SWAY LEFT"
        left = (math.sin(self.yaw_lock), -math.cos(self.yaw_lock))       # body -Y in NED
        v0 = self.pipe.ocr_version
        sp = list(start)
        dt, found, hold_t = 0.05, None, None
        while True:
            meas = self.tick(sp)
            sway = ((self.drone.ned[0] - start[0]) * left[0] + (self.drone.ned[1] - start[1]) * left[1])
            if self.pipe.ocr_version > v0 and meas is not None:
                found = meas
                break
            sp_sway = (sp[0] - start[0]) * left[0] + (sp[1] - start[1]) * left[1]
            if sp_sway < a.max_sway:
                step = min(a.sway_speed * dt, a.max_sway - sp_sway)
                sp[0] += left[0] * step
                sp[1] += left[1] * step
            else:
                hold_t = hold_t or time.time()
                self.state_text = "SEARCH: MAX SWAY, HOLDING"
                if time.time() - hold_t > a.search_timeout:
                    break
            print(f"[SEARCH] sway {sway:4.2f} m / {a.max_sway:.1f} m  (OCR runs {self.pipe.detector.ocr_runs})",
                  end="\r")
            time.sleep(dt)

        if found is None:
            print("\n[RESULT] Banner NOT found within max sway. Landing.")
            self.state_text = "BANNER NOT FOUND - LANDING"
            self.drone.land()
            self.run_for(3.0)
            return

        self.sway_done = ((self.drone.ned[0] - start[0]) * left[0] + (self.drone.ned[1] - start[1]) * left[1])
        bn, be = self.banner_to_ned(found)
        target = [bn, be, -a.alt]
        print(f"\n{'!' * 60}\n[BANNER DETECTED] text='{found['text']}' after swaying {self.sway_done:.2f} m"
              f"{' (partial view, centre reconstructed)' if found['partial'] else ''}\n"
              f"[LOCATION] Drone N={self.drone.ned[0]:+.2f} E={self.drone.ned[1]:+.2f}\n"
              f"[TARGET]   Banner centre estimate N={bn:+.2f} E={be:+.2f}\n{'!' * 60}")

        # STAGE 3 -------------------------------------------------------------
        print(f"\n{'=' * 60}\nSTAGE 3: BANNER ALIGNMENT\n{'=' * 60}")
        print("[BRAKE] Switching to BRAKE to stop immediately...")
        self.state_text = "BANNER DETECTED - BRAKE"
        self.drone.set_mode("BRAKE")
        self.run_for(1.5)

        self.drone.set_mode("GUIDED")
        time.sleep(0.2)
        self.state_text = "BANNER ALIGN: FLY TO ESTIMATE"
        print(f"[GUIDED] Flying over banner estimate N={target[0]:+.2f} E={target[1]:+.2f} ...")
        self.fly_to(target, timeout=15.0)
        target, locked = self.center_over_banner(target, a.center_iters, a.center_tol, f"{a.alt:.0f} m")

        self.state_text = f"ALIGNMENT HOLD ({a.align_duration:.0f}s)"
        t0 = time.time()
        while time.time() - t0 < a.align_duration:
            self.countdown = a.align_duration - (time.time() - t0)
            self.tick(target)
            time.sleep(0.05)
        self.countdown = 0.0
        print(f"[ALIGN] {'Locked' if locked else 'Best effort'} over banner at N={target[0]:+.2f} "
              f"E={target[1]:+.2f}, heading {math.degrees(self.yaw_lock):+.1f} deg")

        # STAGE 4 -------------------------------------------------------------
        print(f"\n{'=' * 60}\nSTAGE 4: DESCEND TO {a.corridor_alt:.1f} m\n{'=' * 60}")
        self.state_text = f"DESCEND TO {a.corridor_alt:.1f}m"
        target[2] = -a.corridor_alt
        self.fly_to(target, timeout=20.0, tol=0.12, speed=a.descend_speed)
        self.run_for(1.0, target)
        if a.recenter_low:
            target, _ = self.center_over_banner(target, 4, a.center_tol * 1.5, f"{a.corridor_alt:.0f} m")
        print(f"[DESCEND] At {self.alt:.2f} m over banner (N={self.drone.ned[0]:+.2f} E={self.drone.ned[1]:+.2f})")

        # STAGE 5 -------------------------------------------------------------
        print(f"\n{'=' * 60}\nSTAGE 5: SURGE {a.surge:.1f} m THROUGH CORRIDOR\n{'=' * 60}")
        self.state_text = f"SURGE {a.surge:.0f}m (CORRIDOR)"
        fwd = (math.cos(self.yaw_lock), math.sin(self.yaw_lock))
        end = [target[0] + fwd[0] * a.surge, target[1] + fwd[1] * a.surge, -a.corridor_alt]
        ok = self.fly_to(end, timeout=a.surge / a.surge_speed + 20.0, tol=0.15, speed=a.surge_speed)
        print(f"[SURGE] {'Reached' if ok else 'Timed out near'} corridor end: "
              f"N={self.drone.ned[0]:+.2f} E={self.drone.ned[1]:+.2f} Alt={self.alt:.2f} m")

        # STAGE 6 -------------------------------------------------------------
        print(f"\n{'=' * 60}\nMISSION COMPLETE  (banner found after {self.sway_done:.2f} m of sway)\n{'=' * 60}")
        self.state_text = "MISSION COMPLETE"
        if a.land_after:
            self.drone.land()
            self.run_for(8.0)
            return
        print("[HOLD] Hovering at corridor end. Press Ctrl+C (or 'q' in the HUD) to land.")
        try:
            while True:
                self.tick(end)
                time.sleep(0.05)
        except KeyboardInterrupt:
            self.drone.land()
            self.run_for(3.0)

    def shutdown(self):
        self.pipe.stop()
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="AeroTHON banner search -> align -> corridor surge")
    p.add_argument("--url", default=None, help="MAVLink URL (default: auto 14550/14551/5762/5763)")
    p.add_argument("--alt", type=float, default=5.0, help="search altitude (m)")
    p.add_argument("--max-sway", type=float, default=5.0, help="max left sway (m)")
    p.add_argument("--sway-speed", type=float, default=0.5, help="sway speed (m/s)")
    p.add_argument("--search-timeout", type=float, default=8.0, help="hover at max sway before giving up (s)")
    p.add_argument("--speed", type=float, default=1.0, help="speed for fly-to moves (m/s)")
    p.add_argument("--center-iters", type=int, default=8, help="visual-servo iterations")
    p.add_argument("--center-tol", type=float, default=12.0, help="centring tolerance (px)")
    p.add_argument("--yaw-tol", type=float, default=2.0, help="heading tolerance (deg)")
    p.add_argument("--settle", type=float, default=1.0, help="settle time per servo iteration (s)")
    p.add_argument("--align-duration", type=float, default=3.0, help="hold time over banner (s)")
    p.add_argument("--corridor-alt", type=float, default=3.0, help="altitude inside corridor (m)")
    p.add_argument("--descend-speed", type=float, default=0.5, help="descent speed (m/s)")
    p.add_argument("--no-recenter-low", dest="recenter_low", action="store_false",
                   help="skip the second centring pass at corridor altitude")
    p.add_argument("--surge", type=float, default=10.0, help="forward distance (m)")
    p.add_argument("--surge-speed", type=float, default=0.8, help="surge speed (m/s)")
    p.add_argument("--land-after", action="store_true", help="land at the corridor end")
    p.add_argument("--image-topic", default="/camera/image_raw")
    p.add_argument("--ocr-threads", type=int, default=2)
    p.add_argument("--verbose", action="store_true", help="detector debug output")
    p.add_argument("--no-camera", dest="show_camera", action="store_false", help="no HUD window")
    args = p.parse_args()

    rclpy.init()
    mission = BannerCorridorMission(args)
    try:
        mission.run()
    except KeyboardInterrupt:
        print("\n[ABORT] Interrupted - landing.")
        if mission.drone is not None:
            mission.drone.land()
    finally:
        mission.shutdown()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        os._exit(0)          # OCR thread may be mid-inference; exit hard like the detector does


if __name__ == "__main__":
    main()
