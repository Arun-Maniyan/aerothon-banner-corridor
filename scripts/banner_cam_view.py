#!/usr/bin/env python3
"""
Live bottom-camera viewer with YOUR banner detector (no flight control).
Shows the banner outline, its four corners (pixel coords) and its centre.

Useful to check the world/camera/detector before running the mission, e.g. while
you fly the drone by hand from the MAVProxy console:
    mode guided ; arm throttle ; takeoff 5 ; then  'position 0 -3 0'  (relative NED move)

Run (with the Gazebo -> ROS bridge running, see run_banner_cam_view.sh):
    python3 banner_cam_view.py [--alt 5.0]
"""
import argparse
import os
import sys
import threading
import time

import cv2
import rclpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import banner_sim_common as bsc  # noqa: E402

WINDOW = "Drone Downward Camera (Banner Viewer)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-topic", default="/camera/image_raw")
    p.add_argument("--alt", type=float, default=5.0,
                   help="altitude used only for the metric offset readout / partial-view centre")
    p.add_argument("--ocr-threads", type=int, default=2)
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()

    rclpy.init()
    node = rclpy.create_node("aerothon_banner_cam_view")
    src = bsc.RosFrameSource(node, a.image_topic)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    pipe = bsc.start_banner_pipeline(src, a.ocr_threads, a.verbose)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 720)
    print("Viewer running. Press 'q' in the window or Ctrl+C to quit.")
    fps, t_prev, last_id = 0.0, None, -1
    try:
        while True:
            frame, fid = src.latest()
            if fid != last_id and frame is not None:
                now = time.time()
                if t_prev:
                    fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
                t_prev, last_id = now, fid
            snap = pipe.state.snapshot()
            meas = bsc.measure_banner(frame, snap, a.alt) if snap else None
            img = bsc.draw_hud(frame, meas, "VIEWER (detector only)",
                               f"cam {fps:4.1f} fps  OCR runs {pipe.detector.ocr_runs}  "
                               f"last OCR {pipe.detector.proc_time:.2f}s",
                               connected=src.connected, alt_rel=a.alt)
            cv2.imshow(WINDOW, img)
            if cv2.waitKey(15) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        os._exit(0)


if __name__ == "__main__":
    main()
