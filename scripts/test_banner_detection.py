#!/usr/bin/env python3
"""
Offline check of the banner detector + world geometry (no Gazebo / ROS needed).

Renders what the repo's downward camera (640x480, HFOV 90 deg, nose = image top)
sees while the drone sways LEFT at --alt, and runs YOUR detector
(siyi_banner_live_rapid_qrcam.detect_banner, RapidOCR) on every frame.
Prints the sway distance at which the banner is first detected.

  python3 test_banner_detection.py                 # default world geometry
  python3 test_banner_detection.py --save-dir /tmp/banner_frames
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import siyi_banner_live_rapid_qrcam as bd  # noqa: E402  (user's detector, unchanged)

TEX = os.path.join(HERE, "..", "src", "aerothon_sim", "models", "aerothon_banner",
                   "materials", "textures", "aerothon_banner.png")
BANNER_L, BANNER_W = 3.5052, 0.3048     # 11.5 ft x 1 ft
IMG_W, IMG_H, HFOV = 640, 480, np.pi / 2
CAM_BELOW_HOME = 0.145                  # camera height = rel. alt + 0.195 (spawn) - 0.05 (mount)


def render(tex, drone_n, drone_e, alt, banner_n, banner_e, rng, ground=(110, 112, 115)):
    f = (IMG_W / 2) / np.tan(HFOV / 2)
    hc = alt + CAM_BELOW_HOME

    def proj(n, e):                      # world (N, E) -> pixel, yaw = 0 (facing North)
        return [IMG_W / 2 + f * (e - drone_e) / hc, IMG_H / 2 - f * (n - drone_n) / hc]

    th, tw = tex.shape[:2]
    src = np.float32([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]])
    dst = np.float32([
        proj(banner_n + BANNER_W / 2, banner_e - BANNER_L / 2),   # texture TL -> NW corner
        proj(banner_n + BANNER_W / 2, banner_e + BANNER_L / 2),   # TR -> NE
        proj(banner_n - BANNER_W / 2, banner_e + BANNER_L / 2),   # BR -> SE
        proj(banner_n - BANNER_W / 2, banner_e - BANNER_L / 2),   # BL -> SW
    ])
    img = np.full((IMG_H, IMG_W, 3), ground, np.uint8)
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(tex, H, (IMG_W, IMG_H), flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full((th, tw), 255, np.uint8), H, (IMG_W, IMG_H))
    img[mask > 127] = warped[mask > 127]
    img = (img.astype(np.float32) * 0.85).clip(0, 255)            # sim shading
    img += rng.normal(0, 2.0, img.shape)                           # sensor noise
    img = cv2.GaussianBlur(img.clip(0, 255).astype(np.uint8), (3, 3), 0)
    return img, dst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alt", type=float, default=5.0)
    p.add_argument("--banner-north", type=float, default=2.0, help="banner centre, m ahead")
    p.add_argument("--banner-east", type=float, default=-7.4, help="banner centre, m East (+) / West (-)")
    p.add_argument("--max-sway", type=float, default=5.0)
    p.add_argument("--step", type=float, default=0.1)
    p.add_argument("--rotate180", action="store_true", help="simulate an upside-down texture")
    p.add_argument("--save-dir", default=None)
    a = p.parse_args()

    tex = cv2.imread(TEX)
    if tex is None:
        sys.exit(f"texture not found: {TEX}")
    if a.rotate180:
        tex = cv2.rotate(tex, cv2.ROTATE_180)
    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)

    reader = bd.build_reader(2)
    rng = np.random.default_rng(0)
    first = None
    for sway in np.arange(0.0, a.max_sway + 1e-6, a.step):
        img, _ = render(tex, 0.0, -sway, a.alt, a.banner_north, a.banner_east, rng)
        best = bd.detect_banner(img, reader)
        if best is None:                       # same 180-deg fallback the mission uses
            best = bd.detect_banner(cv2.rotate(img, cv2.ROTATE_180), reader)
        ok = best is not None
        if ok and first is None:
            first = sway
        if a.save_dir:
            out = img.copy()
            if ok:
                x0, y0, x1, y1 = best["box"]
                cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(a.save_dir, f"sway_{sway:.1f}.png"), out)
        print(f"sway {sway:4.1f} m  -> {'DETECTED  ' + repr(best['text']) + ' box=' + str(best['box']) if ok else '-'}")
    print("\nFIRST DETECTION AT SWAY:", "none" if first is None else f"{first:.1f} m")


if __name__ == "__main__":
    main()
