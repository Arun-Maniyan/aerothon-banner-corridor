"""
AeroTHON 2026 green-banner detector - LIVE version for SIYI A8 mini on Raspberry Pi 5.

CAMERA ACCESS IS IDENTICAL TO THE WORKING QR SCRIPT (siyi_qr_test_node.py):
    cap = cv2.VideoCapture("rtsp://192.168.144.25:8554/main.264", cv2.CAP_FFMPEG)
    a background thread calls cap.read() in a loop and keeps only the newest frame
No OPENCV_FFMPEG_CAPTURE_OPTIONS, no open/read timeout parameters, no CAP_PROP_BUFFERSIZE,
no grab()/retrieve() split and no network pre-check before opening the stream.

Everything else is unchanged from the previous banner script:
  OCR engine: RapidOCR only (PP-OCR models on ONNX Runtime). No EasyOCR / PyTorch.
  pip install rapidocr onnxruntime opencv-python numpy

Threads:
  grabber  : cap.read() loop, keeps only the newest frame (re-opens the stream the
             same way if no frame arrives for a few seconds)
  detector : OCR pipeline, CROP-FIRST (OCR only the green panels). Idle when no green.
  tracker  : follows the green panel every frame between OCR passes (cheap, ~15 Hz)
  stats    : prints a health line every 5 s
  main     : live display (optional) + clean shutdown

Run:
  python -u siyi_banner_live_rapid_qrcam.py 2>&1 | tee ~/Documents/log.txt

Options:
  --stream main|sub | --url RTSP | --source 0|file   --no-display   --save-dir DIR
  --ocr-threads 2   --full-fallback   --quiet-ffmpeg   --verbose
  --check   (only test network/RTSP with a raw socket and exit; never used in a normal run)
"""
import argparse
import os
import re
import sys
import threading
import time
import warnings

# Only the FFmpeg LOG LEVEL is touched (never transport / timeouts), so the stream is
# opened with OpenCV's defaults exactly like the QR script.
if "--quiet-ffmpeg" in sys.argv:
    os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"      # silence HEVC decoder spam

warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np

# =====================================================================
# CONFIG
# =====================================================================
CAMERA_IP = "192.168.144.25"
RTSP_URLS = {
    "main": f"rtsp://{CAMERA_IP}:8554/main.264",
    "sub": f"rtsp://{CAMERA_IP}:8554/sub.264",
}

RECONNECT_AFTER_S = 5.0      # re-open the capture if read() fails for this long

DETECT_MAX_SIDE = 1280       # work-frame size used for the green mask + crops
FULL_OCR_SIDE = 640          # only used by the (rare) full-frame fallback
DETECT_MIN_PERIOD = 0.25     # detector never runs faster than 4 Hz (saves CPU when idle)
OCR_THREADS = 2              # ONNX threads; leaves cores for the video decoder

HOLD_SECONDS = 1.0           # keep OCR result visible this long if tracker has nothing
TRACK_SIDE = 480             # tracker works on a frame this size (cheap)
TRACK_FPS = 15
TRACK_MIN_IOU = 0.25
TRACK_MAX_MISSES = 5         # consecutive frames without the panel -> lock lost
TRACK_MAX_AGE = 6.0          # drop the track if OCR hasn't re-confirmed it for this long

DISPLAY_WIDTH = 960

LOWER_GREEN = np.array([30, 15, 40])
UPPER_GREEN = np.array([85, 255, 255])

TARGET_BRAND = "aerothon"
TARGET_YEAR = "2026"
MAX_BRAND_DIST = 2

MIN_OCR_CONF = 0.15
MIN_GREEN_SUPPORT = 0.55
GROUP_GAP_FACTOR = 1.5
MIN_BANNER_TO_TEXT_AREA = 1.2
MAX_BANNER_FRAC = 0.30

MIN_BANNER_AREA_FRAC = 0.0005   # candidate green blob >= 0.05 % of the work frame
MAX_CROP_CANDIDATES = 2
CROP_TARGET_SIZE = 512          # OCR crops are scaled so the long side is ~this
CROP_MIN_SCALE = 0.5
CROP_MAX_SCALE = 4.0

OCR_KWARGS = dict(
    text_threshold=0.6,
    low_text=0.3,
    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ",
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =====================================================================
# Small helpers
# =====================================================================
def _odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1


def _ellipse(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


_CONFUSION = str.maketrans({
    "o": "0", "q": "0", "d": "0",
    "i": "1", "l": "1", "|": "1",
    "z": "2",
    "s": "6", "5": "6",
    "g": "6", "b": "8",
})


def canon(text):
    text = re.sub(r"[^a-z0-9|]", "", text.lower())
    return text.translate(_CONFUSION)


BRAND_C = canon(TARGET_BRAND)
YEAR_C = canon(TARGET_YEAR)


def partial_match_distance(text, target):
    """Minimum edit distance between `target` and ANY substring of `text`."""
    if len(target) == 0:
        return 0
    if len(text) == 0:
        return len(target)

    previous_row = [0] * (len(text) + 1)
    for i, c_target in enumerate(target):
        current_row = [i + 1]
        for j, c_text in enumerate(text):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c_target != c_text)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return min(previous_row)


def match_target(text):
    c = canon(text)
    return partial_match_distance(c, BRAND_C), (YEAR_C in c)


# =====================================================================
# Detections + grouping
# =====================================================================
def make_detection(poly, text, conf):
    poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    (cx, cy), (w, h), _ = cv2.minAreaRect(poly)
    x, y, bw, bh = cv2.boundingRect(poly.astype(np.int32))
    return {
        "poly": poly,
        "contour": poly.reshape(-1, 1, 2),
        "text": text,
        "conf": float(conf),
        "center": (cx, cy),
        "height": max(1.0, min(w, h)),
        "width": max(w, h),
        "box": (x, y, x + bw, y + bh),
        "green": 0.0,
    }


def poly_gap(d1, d2):
    best = np.inf
    for (x, y) in d1["poly"]:
        best = min(best, -cv2.pointPolygonTest(d2["contour"], (float(x), float(y)), True))
    for (x, y) in d2["poly"]:
        best = min(best, -cv2.pointPolygonTest(d1["contour"], (float(x), float(y)), True))
    return max(0.0, best)


def boxes_are_close(d1, d2):
    h1, h2 = d1["height"], d2["height"]
    if max(h1, h2) / min(h1, h2) > 3.0:
        return False
    return poly_gap(d1, d2) <= GROUP_GAP_FACTOR * (h1 + h2) / 2.0


def group_detections(dets):
    parent = list(range(len(dets)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            if boxes_are_close(dets[i], dets[j]):
                parent[find(i)] = find(j)

    groups = {}
    for i, d in enumerate(dets):
        groups.setdefault(find(i), []).append(d)
    return list(groups.values())


# =====================================================================
# Green analysis
# =====================================================================
def build_green_mask(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def green_support(green_mask, poly, height):
    H, W = green_mask.shape
    pad = max(3, int(0.3 * height))
    pts = np.round(poly).astype(np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    x0, y0 = max(0, x - 2 * pad), max(0, y - 2 * pad)
    x1, y1 = min(W, x + w + 2 * pad), min(H, y + h + 2 * pad)
    if x1 <= x0 or y1 <= y0:
        return 0.0

    roi = green_mask[y0:y1, x0:x1]
    k = _odd(max(3, int(0.6 * height)))
    closed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, _ellipse(k))

    region = np.zeros_like(roi)
    cv2.fillPoly(region, [pts - np.array([x0, y0], dtype=np.int32)], 255)
    region = cv2.dilate(region, _ellipse(_odd(2 * pad + 1)))

    area = cv2.countNonZero(region)
    if area == 0:
        return 0.0
    return cv2.countNonZero(cv2.bitwise_and(closed, region)) / area


def locate_banner(green_mask, group_box, height):
    H, W = green_mask.shape
    gx0, gy0, gx1, gy1 = group_box
    gx0, gy0 = max(0, gx0), max(0, gy0)
    gx1, gy1 = min(W, gx1), min(H, gy1)
    if gx1 <= gx0 or gy1 <= gy0:
        return None

    k = _odd(max(5, int(0.6 * height)))
    closed = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, _ellipse(k))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    sub = labels[gy0:gy1, gx0:gx1]
    ids = sub[sub > 0]
    if ids.size == 0:
        return None
    lab = int(np.bincount(ids.ravel()).argmax())
    bx, by, bw, bh, _ = stats[lab]

    text_area = (gx1 - gx0) * (gy1 - gy0)
    banner_area = bw * bh

    if banner_area > MAX_BANNER_FRAC * H * W:
        return {"box": (gx0, gy0, gx1, gy1), "merged": True}
    if banner_area < MIN_BANNER_TO_TEXT_AREA * text_area:
        return None
    return {"box": (int(bx), int(by), int(bx + bw), int(by + bh)), "merged": False}


def green_candidate_boxes(green_mask):
    """
    Largest, reasonably rectangular green blobs -> padded crop boxes.
    Also returns True if a very large green blob exists (banner may be close-up or
    merged with a green background), which justifies a full-frame OCR fallback.
    """
    H, W = green_mask.shape
    k = _odd(max(5, int(0.01 * max(H, W))))
    closed = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, _ellipse(k))
    n, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    cands, has_large = [], False
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < MIN_BANNER_AREA_FRAC * H * W:
            continue
        if w * h > MAX_BANNER_FRAC * H * W:
            has_large = True
            continue
        if a / float(w * h) < 0.5:
            continue
        cands.append((a, x, y, w, h))
    cands.sort(reverse=True)

    boxes = []
    for a, x, y, w, h in cands[:MAX_CROP_CANDIDATES]:
        px, py = int(0.1 * w), int(0.1 * h)
        boxes.append((max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py)))
    return boxes, has_large


# =====================================================================
# OCR
# =====================================================================
def run_ocr(reader, img, roi=None, scale=1.0):
    """OCR the full image or a crop; polygons are returned in FULL-image coordinates."""
    ox, oy = 0, 0
    src = img
    if roi is not None:
        x0, y0, x1, y1 = roi
        src = img[y0:y1, x0:x1]
        ox, oy = x0, y0
    if scale != 1.0:
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        src = cv2.resize(src, None, fx=scale, fy=scale, interpolation=interp)

    dets = []
    for bbox, text, conf in reader.readtext(src, **OCR_KWARGS):
        poly = np.asarray(bbox, dtype=np.float32) / scale + np.array([ox, oy], dtype=np.float32)
        dets.append(make_detection(poly, text, conf))
    return dets


# =====================================================================
# Core detection logic
# =====================================================================
def find_banner(img, green_mask, detections, verbose=False):
    """Returns (best_result_or_None, rejected_detections)."""
    H, W = img.shape[:2]
    on_green, rejected = [], []

    for d in detections:
        d["green"] = green_support(green_mask, d["poly"], d["height"])
        if d["conf"] >= MIN_OCR_CONF and d["green"] >= MIN_GREEN_SUPPORT:
            on_green.append(d)
        else:
            rejected.append(d)

    candidates = []
    for gi, group in enumerate(group_detections(on_green), 1):
        group.sort(key=lambda d: (round(d["center"][1] / d["height"]), d["center"][0]))
        text = " ".join(d["text"] for d in group)
        brand_dist, year_ok = match_target(text)
        mean_green = float(np.mean([d["green"] for d in group]))
        mean_conf = float(np.mean([d["conf"] for d in group]))

        pts = np.vstack([d["poly"] for d in group])
        gx, gy, gw, gh = cv2.boundingRect(np.round(pts).astype(np.int32))
        group_box = (gx, gy, gx + gw, gy + gh)
        height = float(np.median([d["height"] for d in group]))

        if verbose:
            log(f"  group [{gi}] \"{text}\" brand_dist={brand_dist} year={year_ok} "
                f"green={mean_green*100:.0f}%")

        if brand_dist > MAX_BRAND_DIST or not year_ok:
            rejected.extend(group)
            continue

        banner = locate_banner(green_mask, group_box, height)
        if banner is None:
            rejected.extend(group)
            continue

        x0, y0, x1, y1 = banner["box"]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        candidates.append({
            "text": text,
            "brand_dist": brand_dist,
            "green": mean_green,
            "conf": mean_conf,
            "box": banner["box"],
            "text_box": group_box,
            "center": (int(round(cx)), int(round(cy))),
            "merged_with_background": banner["merged"],
        })

    if not candidates:
        return None, rejected
    candidates.sort(key=lambda c: (c["brand_dist"], -c["green"], -c["conf"]))
    return candidates[0], rejected


def _center_inside(box, outer):
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def detect_banner(img, reader, hint_box=None, full_fallback=False, verbose=False):
    """
    CROP-FIRST detection. `img` is the (downscaled) work frame.
    hint_box (work-frame coords) = where the tracker believes the banner is; it is
    OCR'd first so a tracked banner is re-verified in about a second.
    Returns best or None.
    """
    H, W = img.shape[:2]
    green_mask = build_green_mask(img)
    boxes, has_large = green_candidate_boxes(green_mask)

    if hint_box is not None:
        hx0, hy0, hx1, hy1 = hint_box
        px, py = int(0.15 * (hx1 - hx0)), int(0.15 * (hy1 - hy0))
        hb = (max(0, hx0 - px), max(0, hy0 - py), min(W, hx1 + px), min(H, hy1 + py))
        if hb[2] > hb[0] and hb[3] > hb[1]:
            boxes = [hb] + [b for b in boxes if not _center_inside(b, hb)]

    for (x0, y0, x1, y1) in boxes[:MAX_CROP_CANDIDATES + 1]:
        scale = float(np.clip(CROP_TARGET_SIZE / float(max(x1 - x0, y1 - y0)),
                              CROP_MIN_SCALE, CROP_MAX_SCALE))
        dets = run_ocr(reader, img, roi=(x0, y0, x1, y1), scale=scale)
        best, _ = find_banner(img, green_mask, dets, verbose)
        if best is not None:
            return best

    if full_fallback or has_large:
        scale = min(1.0, FULL_OCR_SIDE / float(max(H, W)))
        best, _ = find_banner(img, green_mask, run_ocr(reader, img, scale=scale), verbose)
        return best
    return None


# =====================================================================
# Shared state + panel tracker
# =====================================================================
class BannerState:
    """Latest banner box in FULL-frame coordinates, from OCR or from the tracker."""

    def __init__(self):
        self.lock = threading.Lock()
        self.box = None
        self.text = ""
        self.source = ""
        self.t_ocr = 0.0
        self.ocr_version = 0
        self.size = (1, 1)        # (W, H) of the full frame

    def set_ocr(self, box, text, size):
        with self.lock:
            self.box, self.text, self.source = tuple(box), text, "OCR"
            self.t_ocr = time.time()
            self.ocr_version += 1
            self.size = size

    def set_track(self, box, size):
        with self.lock:
            if self.box is not None:
                self.box, self.source, self.size = tuple(box), "TRACK", size

    def clear(self):
        with self.lock:
            self.box = None
            self.source = ""

    def get_box(self):
        with self.lock:
            return self.box

    def snapshot(self):
        with self.lock:
            if self.box is None:
                return None
            W, H = self.size
            x0, y0, x1, y1 = self.box
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            return {
                "box": self.box, "text": self.text, "source": self.source,
                "center": (int(round(cx)), int(round(cy))),
                "dx_norm": (cx - W / 2.0) / (W / 2.0),
                "dy_norm": (cy - H / 2.0) / (H / 2.0),
                "ocr_age": time.time() - self.t_ocr,
            }


class PanelTracker:
    """Follows the green panel between OCR passes by re-finding the green blob that
    overlaps the previous box. Very cheap (works on a ~480 px frame)."""

    def __init__(self):
        self.box = None
        self.misses = 0

    def init(self, box):
        self.box = tuple(box)
        self.misses = 0

    def update(self, frame):
        if self.box is None:
            return None
        H, W = frame.shape[:2]
        s = min(1.0, TRACK_SIDE / float(max(H, W)))
        small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else frame
        sh, sw = small.shape[:2]
        mask = build_green_mask(small)
        k = _odd(max(5, int(0.02 * max(sh, sw))))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(k))
        n, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

        bx0, by0, bx1, by1 = [v * s for v in self.box]
        barea = max(1.0, (bx1 - bx0) * (by1 - by0))
        best, best_iou = None, TRACK_MIN_IOU
        for i in range(1, n):
            x, y, w, h, _a = stats[i]
            if w * h < 0.3 * barea or w * h > 3.0 * barea:
                continue
            inter = max(0.0, min(x + w, bx1) - max(x, bx0)) * max(0.0, min(y + h, by1) - max(y, by0))
            iou = inter / (w * h + barea - inter)
            if iou > best_iou:
                best_iou, best = iou, (x, y, x + w, y + h)
        if best is None:
            self.misses += 1
            return None
        self.misses = 0
        self.box = tuple(int(round(v / s)) for v in best)
        return self.box


# =====================================================================
# Optional network check (ONLY used with --check, never in a normal run)
# =====================================================================
def _local_ipv4():
    import subprocess
    ips = []
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr"], capture_output=True, text=True, timeout=2).stdout
        for line in out.splitlines():
            parts = line.split()
            if "inet" in parts:
                ips.append((parts[1], parts[parts.index("inet") + 1].split("/")[0]))
    except Exception:
        pass
    return ips


def preflight(url):
    """Checks Pi IP / camera reachability / RTSP reply. Returns True if RTSP answers."""
    import socket
    from urllib.parse import urlparse
    u = urlparse(url)
    host, port = u.hostname, u.port or 554
    ips = _local_ipv4()
    subnet = ".".join(host.split(".")[:3]) + "."
    same = [(n, ip) for n, ip in ips if ip.startswith(subnet)]
    log("[check] Pi addresses: " + (", ".join(f"{n}={ip}" for n, ip in ips) or "unknown"))
    if ips and not same and not host.startswith("127."):
        log(f"[check] PROBLEM: no Pi interface is on {subnet}x (give the Ethernet port e.g. {subnet}12/24)")
        return False
    try:
        sock = socket.create_connection((host, port), timeout=3)
    except Exception as e:
        log(f"[check] PROBLEM: cannot reach {host}:{port} ({e})")
        return False
    try:
        sock.settimeout(3)
        req = f"DESCRIBE {url} RTSP/1.0\r\nCSeq: 1\r\nAccept: application/sdp\r\n\r\n"
        sock.sendall(req.encode())
        reply = sock.recv(512).decode(errors="replace").splitlines()
        first = reply[0] if reply else "(empty reply)"
    except Exception as e:
        log(f"[check] PROBLEM: {host}:{port} accepts connections but RTSP does not answer ({e})")
        return False
    finally:
        sock.close()
    ok = "200" in first
    log(f"[check] RTSP reply: {first}")
    return ok


# =====================================================================
# Camera + threads
# =====================================================================
def open_capture(source):
    """Open the camera EXACTLY like the working QR script does."""
    if isinstance(source, int):
        return cv2.VideoCapture(source)
    return cv2.VideoCapture(source, cv2.CAP_FFMPEG)


class FrameGrabber(threading.Thread):
    """Same loop as the QR script's _capture_loop: cap.read() forever, keep the newest
    frame, sleep 10 ms when a read fails. If reads keep failing for RECONNECT_AFTER_S the
    capture is re-opened with the very same call."""

    def __init__(self, source, cap):
        super().__init__(daemon=True)
        self.source = source
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.running = True
        self.connected = False
        self.grabbed = 0
        self.reconnects = 0
        self.last_frame_time = 0.0

    def run(self):
        fail_since = None
        while self.running:
            if self.cap is None:
                log(f"[grabber] reconnecting to {self.source} ...")
                self.cap = open_capture(self.source)
                if not self.cap.isOpened():
                    self.cap.release()
                    self.cap = None
                    log("[grabber] could not open stream, retry in 2 s")
                    time.sleep(2.0)
                    continue
                log("[grabber] stream opened")

            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.connected = False
                now = time.time()
                if fail_since is None:
                    fail_since = now
                elif now - fail_since > RECONNECT_AFTER_S:
                    self.reconnects += 1
                    log(f"[grabber] no frame for {RECONNECT_AFTER_S:.0f}s (reconnect #{self.reconnects})")
                    self.cap.release()
                    self.cap = None
                    fail_since = None
                    time.sleep(0.5)
                    continue
                time.sleep(0.01)
                continue

            fail_since = None
            self.connected = True
            self.grabbed += 1
            self.last_frame_time = time.time()
            with self.lock:
                self.frame = frame              # read() returns a fresh array every call
                self.frame_id += 1

        if self.cap is not None:
            self.cap.release()

    def latest(self):
        with self.lock:
            return self.frame, self.frame_id

    def stop(self):
        self.running = False


class DetectorWorker(threading.Thread):
    def __init__(self, grabber, reader, state, full_fallback=False, verbose=False):
        super().__init__(daemon=True)
        self.grabber, self.reader, self.state = grabber, reader, state
        self.full_fallback, self.verbose = full_fallback, verbose
        self.running = True
        self.proc_time = 0.0
        self.ocr_runs = 0
        self.on_detect = None

    def run(self):
        last_id = 0
        while self.running:
            frame, fid = self.grabber.latest()
            if frame is None or fid == last_id:
                time.sleep(0.01)
                continue
            last_id = fid

            t0 = time.time()
            H, W = frame.shape[:2]
            scale = min(1.0, DETECT_MAX_SIDE / float(max(H, W)))
            work = cv2.resize(frame, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA) if scale < 1.0 else frame

            hint = self.state.get_box()
            if hint is not None:
                hint = tuple(int(v * scale) for v in hint)
            try:
                best = detect_banner(work, self.reader, hint, self.full_fallback, self.verbose)
            except Exception as e:                 # never let OCR kill the thread
                log(f"[detector] error: {e}")
                best = None

            self.proc_time = time.time() - t0
            self.ocr_runs += 1
            if best is not None:
                box = tuple(int(round(v / scale)) for v in best["box"])
                self.state.set_ocr(box, best["text"], (W, H))
                snap = self.state.snapshot()
                log(f"[detector] BANNER center={snap['center']} "
                    f"offset(dx,dy)=({snap['dx_norm']:+.2f},{snap['dy_norm']:+.2f}) "
                    f"text=\"{best['text']}\" ({self.proc_time:.2f}s)")
                if self.on_detect:
                    self.on_detect(frame)
            elif self.verbose:
                log(f"[detector] nothing ({self.proc_time:.2f}s)")

            time.sleep(max(0.0, DETECT_MIN_PERIOD - (time.time() - t0)))

    def stop(self):
        self.running = False


class TrackerWorker(threading.Thread):
    def __init__(self, grabber, state):
        super().__init__(daemon=True)
        self.grabber, self.state = grabber, state
        self.tracker = PanelTracker()
        self.running = True
        self.seen_version = -1
        self.last_log = 0.0

    def run(self):
        last_id = 0
        period = 1.0 / TRACK_FPS
        while self.running:
            t0 = time.time()
            frame, fid = self.grabber.latest()
            box = self.state.get_box()
            if frame is None or fid == last_id or box is None:
                time.sleep(0.02)
                continue
            last_id = fid

            with self.state.lock:
                ver, age = self.state.ocr_version, time.time() - self.state.t_ocr
            if ver != self.seen_version:             # fresh OCR result -> re-lock tracker
                self.seen_version = ver
                self.tracker.init(box)
            if age > TRACK_MAX_AGE:
                self.state.clear()
                log("[track] expired (no OCR confirmation)")
                continue

            H, W = frame.shape[:2]
            new_box = self.tracker.update(frame)
            if new_box is not None:
                self.state.set_track(new_box, (W, H))
                if time.time() - self.last_log > 1.0:
                    self.last_log = time.time()
                    s = self.state.snapshot()
                    if s:
                        log(f"[track] center={s['center']} "
                            f"offset=({s['dx_norm']:+.2f},{s['dy_norm']:+.2f})")
            elif self.tracker.misses > TRACK_MAX_MISSES:
                self.state.clear()
                log("[track] lock lost")
            time.sleep(max(0.0, period - (time.time() - t0)))

    def stop(self):
        self.running = False


def _read_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return float("nan")


def stats_loop(grabber, worker, state, stop_evt, period=5.0):
    pg, pt = grabber.grabbed, time.time()
    while not stop_evt.wait(period):
        now = time.time()
        dt = now - pt
        g = grabber.grabbed
        age = now - grabber.last_frame_time if grabber.last_frame_time else float("inf")
        s = state.snapshot()
        st = f"{s['source']}@{s['center']}" if s else "none"
        try:
            load = os.getloadavg()[0]
        except Exception:
            load = float("nan")
        log(f"[stats] read={(g-pg)/dt:.1f}fps reconnects={grabber.reconnects} "
            f"last_frame_age={age:.1f}s ocr_last={worker.proc_time:.2f}s "
            f"ocr_runs={worker.ocr_runs} banner={st} load1={load:.2f} temp={_read_temp():.0f}C")
        pg, pt = g, now


# =====================================================================
# Visualisation
# =====================================================================
def _label(img, text, org, colour, scale=0.5):
    """Small text with a dark outline so it stays readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def draw_result(img, snap, scale, connected, fps=None):
    """img is already resized by `scale`; snap boxes are in full-frame coords."""
    out = img
    H, W = out.shape[:2]
    cv2.drawMarker(out, (W // 2, H // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
    if snap is not None:
        x0, y0, x1, y1 = [int(v * scale) for v in snap["box"]]
        c = (int(snap["center"][0] * scale), int(snap["center"][1] * scale))
        colour = (0, 255, 0) if snap["source"] == "OCR" else (0, 200, 255)
        cv2.rectangle(out, (x0, y0), (x1, y1), colour, 3)
        cv2.circle(out, c, 8, colour, -1)
        cv2.line(out, (W // 2, H // 2), c, (0, 255, 255), 2)
        cv2.putText(out, f"AeroTHON 2026 BANNER [{snap['source']}]", (x0, max(20, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)

        # Bounding-box corners (pixel coordinates in the full camera frame)
        fx0, fy0, fx1, fy1 = snap["box"]
        corner_col = (255, 0, 255)                       # magenta
        for (px, py), (fx, fy), dx, dy in (
            ((x0, y0), (fx0, fy0), 10, 22),              # top-left
            ((x1, y0), (fx1, fy0), -96, 22),             # top-right
            ((x1, y1), (fx1, fy1), -96, -12),            # bottom-right
            ((x0, y1), (fx0, fy1), 10, -12),             # bottom-left
        ):
            cv2.circle(out, (px, py), 7, corner_col, -1)
            cv2.circle(out, (px, py), 7, (255, 255, 255), 1)
            _label(out, f"({fx},{fy})", (px + dx, py + dy), corner_col)
        # Centre coordinates
        _label(out, f"C({snap['center'][0]},{snap['center'][1]})", (c[0] + 14, c[1] + 24), (0, 255, 255), 0.55)
    if not connected:
        cv2.putText(out, "NO SIGNAL", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2, cv2.LINE_AA)
    if fps is not None:
        cv2.putText(out, f"FPS: {fps:4.1f}", (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, f"FPS: {fps:4.1f}", (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
    return out


# =====================================================================
# Driver
# =====================================================================
class RapidReader:
    """Gives RapidOCR a readtext() -> [(4-point polygon, text, confidence)]."""

    def __init__(self, threads):
        import logging
        try:
            from rapidocr import RapidOCR              # current package: pip install rapidocr
            self.new_api = True
            params = {
                "Global.use_cls": False,                # skip the angle classifier (saves time)
                "Det.limit_type": "max",                # never upscale crops inside the detector
                "Det.limit_side_len": 640,
                "EngineConfig.onnxruntime.intra_op_num_threads": threads,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            }
            try:
                self.engine = RapidOCR(params=params)
            except Exception as e:
                log(f"RapidOCR custom params rejected ({e}); using defaults")
                self.engine = RapidOCR()
        except ImportError:
            try:
                from rapidocr_onnxruntime import RapidOCR   # older package
            except ImportError:
                raise SystemExit("RapidOCR not installed. Run:  pip install rapidocr onnxruntime")
            self.new_api = False
            try:
                self.engine = RapidOCR(use_cls=False, intra_op_num_threads=threads)
            except TypeError:
                self.engine = RapidOCR()
        logging.getLogger("RapidOCR").setLevel(logging.ERROR)

    def readtext(self, img, **_ignored):
        out = []
        try:
            if self.new_api:
                r = self.engine(img, use_cls=False)
                if r is None or r.boxes is None or r.txts is None:
                    return out
                for box, txt, sc in zip(r.boxes, r.txts, r.scores):
                    out.append((np.asarray(box, dtype=np.float32), str(txt), float(sc)))
            else:
                res, _ = self.engine(img, use_cls=False)
                for box, txt, sc in (res or []):
                    out.append((np.asarray(box, dtype=np.float32), str(txt), float(sc)))
        except Exception as e:
            log(f"[ocr] error: {e}")
        return out


def build_reader(ocr_threads):
    log(f"Initializing RapidOCR (onnx threads={ocr_threads}) ...")
    reader = RapidReader(ocr_threads)
    reader.readtext(np.full((64, 256, 3), 128, np.uint8))     # warm-up
    return reader


def parse_args():
    p = argparse.ArgumentParser(description="Live AeroTHON 2026 banner detector (SIYI A8 mini)")
    p.add_argument("--stream", choices=["main", "sub"], default="main")
    p.add_argument("--url", default=None, help="custom RTSP URL")
    p.add_argument("--source", default=None, help="OpenCV source instead of RTSP (0, file ...)")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--save-dir", default=None, help="save annotated frame on detections")
    p.add_argument("--ocr-threads", type=int, default=OCR_THREADS)
    p.add_argument("--full-fallback", action="store_true",
                   help="also OCR the whole frame when green crops fail (slow)")
    p.add_argument("--quiet-ffmpeg", action="store_true", help="silence FFmpeg decoder messages")
    p.add_argument("--check", action="store_true", help="only test the network / RTSP and exit")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cv2.setNumThreads(2)

    if args.source is not None:
        source = int(args.source) if args.source.isdigit() else args.source
    else:
        source = args.url if args.url else RTSP_URLS[args.stream]

    if args.check:
        if isinstance(source, str) and source.startswith("rtsp"):
            raise SystemExit(0 if preflight(source) else 1)
        raise SystemExit("--check only applies to RTSP sources")

    show = not args.no_display
    if show and not os.environ.get("DISPLAY") and os.name != "nt":
        log("No DISPLAY found - running headless.")
        show = False
    if show:
        try:
            cv2.namedWindow("AeroTHON banner (live)")
        except cv2.error:
            log("OpenCV has no GUI support - running headless.")
            show = False
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # Load OCR first (slow), then open the camera and start reading immediately,
    # in the same order the QR script uses (detector first, camera second).
    reader = build_reader(args.ocr_threads)
    state = BannerState()
    stop_evt = threading.Event()

    log(f"Connecting to SIYI camera: {source}")
    cap = open_capture(source)
    if not cap.isOpened():
        log("Could not open stream. Check Ethernet, camera IP and RTSP URL "
            "(run with --check for a network diagnosis).")
        raise SystemExit(1)
    log("Camera connected.")

    grabber = FrameGrabber(source, cap)
    grabber.start()
    worker = DetectorWorker(grabber, reader, state, args.full_fallback, args.verbose)
    tracker = TrackerWorker(grabber, state)

    last_save = [0.0]

    def on_detect(frame):
        if args.save_dir and time.time() - last_save[0] > 1.0:
            last_save[0] = time.time()
            img = frame.copy()
            draw_result(img, state.snapshot(), 1.0, True)
            cv2.imwrite(os.path.join(args.save_dir, f"det_{int(time.time()*1000)}.jpg"), img)

    worker.on_detect = on_detect
    worker.start()
    tracker.start()
    threading.Thread(target=stats_loop, args=(grabber, worker, state, stop_evt),
                     daemon=True).start()

    log("Running. Press 'q' in the window or Ctrl+C to quit.")
    last_shown = 0
    fps_disp, t_prev = 0.0, None          # smoothed FPS of the live feed
    try:
        while True:
            frame, fid = grabber.latest()
            if not show:
                time.sleep(0.1)
                continue
            if frame is None or fid == last_shown:
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                continue
            last_shown = fid
            now = time.time()
            if t_prev is not None and now > t_prev:
                inst = 1.0 / (now - t_prev)
                fps_disp = inst if fps_disp == 0.0 else 0.9 * fps_disp + 0.1 * inst
            t_prev = now
            H, W = frame.shape[:2]
            s = min(1.0, DISPLAY_WIDTH / float(W))
            img = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else frame.copy()
            img = draw_result(img, state.snapshot(), s, grabber.connected, fps_disp)
            cv2.imshow("AeroTHON banner (live)", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        log("Shutting down ...")
        stop_evt.set()
        worker.stop()
        tracker.stop()
        grabber.stop()
        if show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        sys.stdout.flush()
        sys.stderr.flush()
        # The OCR thread may be inside a long OCR call; exit hard to avoid the
        # "terminate called without an active exception" abort.
        os._exit(0)


if __name__ == "__main__":
    main()
