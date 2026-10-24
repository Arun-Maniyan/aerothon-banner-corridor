#!/usr/bin/env python3
"""
Shared glue between the Gazebo downward camera (ROS 2 /camera/image_raw) and YOUR
banner detector (siyi_banner_live_rapid_qrcam.py, used unchanged as a module).

Provides:
  RosFrameSource      - drop-in replacement for the detector's FrameGrabber
                        (same .latest() -> (frame, frame_id) interface)
  start_banner_pipeline - starts the detector's own DetectorWorker + TrackerWorker
  measure_banner      - centre, 4 true corners, angle, partial-view flag from a frame
  pixel_to_ned_offset - repo's pinhole projection (pixel -> North/East metres)
  draw_hud            - live feed overlay: crosshair, banner quad, 4 corners, centre
"""
import math
import os
import sys
import threading
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import siyi_banner_live_rapid_qrcam as bd  # noqa: E402  (user's detector, unchanged file)

# --- camera + banner geometry (repo camera: 640x480, HFOV 90 deg, 5 cm below base_link) ---
HFOV = 1.5707963
CAM_OFFSET_M = 0.145          # camera height above ground = relative alt + 0.195 - 0.05
BANNER_LEN_M = 3.5052         # 11.5 ft
BANNER_WID_M = 0.3048         # 1 ft

# ----------------------------------------------------------------------------------
# Quieter logging: the detector prints every OCR hit (4 Hz); keep max 1 line/s per tag
# ----------------------------------------------------------------------------------
_last_log = {}
_orig_log = bd.log


def _rate_limited_log(msg):
    tag = msg.split("]")[0] if msg.startswith("[") else ""
    if tag in ("[detector", "[track"):
        now = time.time()
        if now - _last_log.get(tag, 0.0) < 1.0:
            return
        _last_log[tag] = now
    _orig_log(msg)


bd.log = _rate_limited_log

# ----------------------------------------------------------------------------------
# 180-degree fallback: if the texture ever renders upside-down in your Gazebo build,
# OCR (run without the angle classifier) would miss it. Retry on the rotated frame.
# ----------------------------------------------------------------------------------
_orig_detect = bd.detect_banner


def _detect_banner_any_orientation(img, reader, hint_box=None, full_fallback=False, verbose=False):
    best = _orig_detect(img, reader, hint_box, full_fallback, verbose)
    if best is not None:
        return best
    H, W = img.shape[:2]
    r = _orig_detect(cv2.rotate(img, cv2.ROTATE_180), reader, None, full_fallback, verbose)
    if r is None:
        return None

    def flip(b):
        x0, y0, x1, y1 = b
        return (W - x1, H - y1, W - x0, H - y0)

    r["box"] = flip(r["box"])
    r["text_box"] = flip(r["text_box"])
    r["center"] = (W - r["center"][0], H - r["center"][1])
    return r


bd.detect_banner = _detect_banner_any_orientation


# ----------------------------------------------------------------------------------
# ROS 2 frame source (same interface as the detector's FrameGrabber)
# ----------------------------------------------------------------------------------
class RosFrameSource:
    def __init__(self, node, topic="/camera/image_raw"):
        from sensor_msgs.msg import Image
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.grabbed = 0
        self.reconnects = 0
        self.last_frame_time = 0.0
        node.create_subscription(Image, topic, self._cb, 10)

    def _cb(self, msg):
        im = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
        if msg.encoding in ("rgb8", "RGB8"):
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        elif msg.encoding in ("rgba8", "RGBA8"):
            im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
        else:
            im = im[:, :, :3].copy()
        with self.lock:
            self.frame = im
            self.frame_id += 1
        self.grabbed += 1
        self.last_frame_time = time.time()

    def latest(self):
        with self.lock:
            return self.frame, self.frame_id

    @property
    def connected(self):
        return time.time() - self.last_frame_time < 1.0

    def stop(self):
        pass


class BannerPipeline:
    """Your detector's DetectorWorker (RapidOCR, ~4 Hz) + TrackerWorker (~15 Hz)."""

    def __init__(self, source, ocr_threads=2, verbose=False):
        self.source = source
        self.reader = bd.build_reader(ocr_threads)
        self.state = bd.BannerState()
        self.detector = bd.DetectorWorker(source, self.reader, self.state, False, verbose)
        self.tracker = bd.TrackerWorker(source, self.state)

    def start(self):
        self.detector.start()
        self.tracker.start()
        return self

    def stop(self):
        self.detector.stop()
        self.tracker.stop()

    @property
    def ocr_version(self):
        with self.state.lock:
            return self.state.ocr_version


def start_banner_pipeline(source, ocr_threads=2, verbose=False):
    return BannerPipeline(source, ocr_threads, verbose).start()


# ----------------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------------
def order_corners(pts):
    """TL, TR, BR, BL (same helper as the repo's QR code)."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    c = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def metres_per_pixel(alt_rel, img_w):
    hc = max(alt_rel + CAM_OFFSET_M, 0.5)
    return 2.0 * hc * math.tan(HFOV / 2.0) / img_w


def measure_banner(frame, snap, alt_rel):
    """
    Returns dict(u, v, quad[4x2 TL,TR,BR,BL], angle_rad, partial, source, text) or None.
    u, v = banner centre in pixels. If the banner is cut by the image border, the centre
    is reconstructed from the visible end + the known 11.5 ft length.
    angle_rad = direction of the banner's long side in the image (0 = horizontal).
    """
    if frame is None or snap is None:
        return None
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in snap["box"]]
    pad = max(4, int(0.15 * max(x1 - x0, y1 - y0)))
    rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
    rx1, ry1 = min(W, x1 + pad), min(H, y1 + pad)
    if rx1 - rx0 < 4 or ry1 - ry0 < 4:
        return None

    roi = frame[ry0:ry1, rx0:rx1]
    mask = bd.build_green_mask(roi)
    k = bd._odd(max(5, int(0.5 * min(x1 - x0, y1 - y0))))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, bd._ellipse(k))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        quad = cv2.boxPoints(cv2.minAreaRect(c)) + np.float32([rx0, ry0])
    else:                                            # fall back to the detector's box
        quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    quad = order_corners(quad)

    # long-side angle (undirected, wrapped to [-90, 90) deg)
    e1, e2 = quad[1] - quad[0], quad[2] - quad[1]
    long_edge = e1 if np.hypot(*e1) >= np.hypot(*e2) else e2
    ang = math.atan2(float(long_edge[1]), float(long_edge[0]))
    if ang >= math.pi / 2:
        ang -= math.pi
    elif ang < -math.pi / 2:
        ang += math.pi

    bx0, by0 = quad[:, 0].min(), quad[:, 1].min()
    bx1, by1 = quad[:, 0].max(), quad[:, 1].max()
    m = 2
    cut_l, cut_r = bx0 <= m, bx1 >= W - 1 - m
    cut_t, cut_b = by0 <= m, by1 >= H - 1 - m
    partial = cut_l or cut_r or cut_t or cut_b

    u, v = float(quad[:, 0].mean()), float(quad[:, 1].mean())
    mpp = metres_per_pixel(alt_rel, W)
    L_px, W_px = BANNER_LEN_M / mpp, BANNER_WID_M / mpp
    if cut_l and not cut_r:
        u = bx1 - L_px / 2.0
    elif cut_r and not cut_l:
        u = bx0 + L_px / 2.0
    if cut_t and not cut_b:
        v = by1 - W_px / 2.0
    elif cut_b and not cut_t:
        v = by0 + W_px / 2.0

    return {"u": u, "v": v, "quad": quad, "angle": ang, "partial": partial,
            "source": snap["source"], "text": snap.get("text", ""), "size": (W, H)}


def pixel_to_ned_offset(u, v, img_w, img_h, alt_rel, yaw):
    """Repo's pinhole projection: image top = nose, image right = right wing."""
    mpp = metres_per_pixel(alt_rel, img_w)
    dx_b = -(v - img_h / 2.0) * mpp          # forward
    dy_b = (u - img_w / 2.0) * mpp           # right
    dn = dx_b * math.cos(yaw) - dy_b * math.sin(yaw)
    de = dx_b * math.sin(yaw) + dy_b * math.cos(yaw)
    return dn, de


# ----------------------------------------------------------------------------------
# Live HUD
# ----------------------------------------------------------------------------------
def _txt(img, text, org, colour, scale=0.5, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def draw_hud(frame, meas, state_text="", telemetry="", countdown=0.0, connected=True,
             alt_rel=None, scale=1.5):
    """Returns an annotated, upscaled copy of the frame."""
    if frame is None:
        img = np.zeros((int(480 * scale), int(640 * scale), 3), np.uint8)
        _txt(img, "WAITING FOR /camera/image_raw ...", (20, img.shape[0] // 2), (0, 0, 255), 0.8, 2)
        return img
    img = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    H, W = img.shape[:2]
    cx, cy = W // 2, H // 2
    cv2.drawMarker(img, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 26, 2)
    cv2.circle(img, (cx, cy), 18, (0, 255, 255), 1)

    if meas is not None:
        colour = (0, 255, 0) if meas["source"] == "OCR" else (0, 200, 255)
        q = (meas["quad"] * scale).astype(np.int32)
        cv2.polylines(img, [q], True, colour, 3)
        names = ("TL", "TR", "BR", "BL")
        offs = ((8, -10), (-120, -10), (-120, 24), (8, 24))
        for (px, py), (fx, fy), name, (dx, dy) in zip(q, meas["quad"], names, offs):
            cv2.circle(img, (int(px), int(py)), 7, (255, 0, 255), -1)
            cv2.circle(img, (int(px), int(py)), 7, (255, 255, 255), 1)
            _txt(img, f"{name}({int(fx)},{int(fy)})", (int(px) + dx, int(py) + dy), (255, 0, 255), 0.5)
        c = (int(meas["u"] * scale), int(meas["v"] * scale))
        cv2.circle(img, c, 9, (0, 0, 255), -1)
        cv2.circle(img, c, 9, (255, 255, 255), 1)
        cv2.line(img, (cx, cy), c, (255, 255, 0), 2)
        _txt(img, f"C({int(meas['u'])},{int(meas['v'])})", (c[0] + 14, c[1] + 26), (0, 255, 255), 0.55)
        tag = f"AeroTHON 2026 BANNER [{meas['source']}{' PARTIAL' if meas['partial'] else ''}]"
        _txt(img, tag, (max(5, int(q[:, 0].min())), max(60, int(q[:, 1].min()) - 34)), colour, 0.6, 2)
        if alt_rel is not None:
            fw, fh = meas["size"]
            dn, de = pixel_to_ned_offset(meas["u"], meas["v"], fw, fh, alt_rel, 0.0)
            _txt(img, f"offset fwd {dn:+.2f} m  right {de:+.2f} m  angle {math.degrees(meas['angle']):+.1f} deg",
                 (10, H - 66), (255, 255, 0), 0.6, 1)

    cv2.rectangle(img, (0, 0), (W, 44), (25, 25, 25), -1)
    _txt(img, f"MISSION: {state_text}", (10, 29), (0, 255, 255), 0.7, 2)
    if telemetry:
        (tw, _), _ = cv2.getTextSize(telemetry, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        _txt(img, telemetry, (max(330, W - tw - 12), 29), (255, 255, 255), 0.5, 1)
    if countdown > 0:
        cv2.rectangle(img, (cx - 190, H - 52), (cx + 190, H - 10), (0, 90, 0), -1)
        _txt(img, f"ALIGNMENT HOLD: {countdown:.1f}s", (cx - 175, H - 22), (0, 255, 0), 0.8, 2)
    if not connected:
        _txt(img, "NO SIGNAL", (10, 80), (0, 0, 255), 0.9, 2)
    return img
