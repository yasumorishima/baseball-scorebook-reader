#!/usr/bin/env python3
"""Grid lattice detection for Seibido 9104 scorebook scans.

Approach: no hand-tuned layout ratios. Extract printed grid lines directly
via morphological opening (long horizontal / vertical structuring elements),
then cluster line positions from connected components. Emits:
  - overlay PNG (downscaled) with detected H lines (green) / V lines (red)
  - JSON with clustered line positions and spacing diagnostics

Usage: grid_detect.py <image> <outdir>
"""
import cv2
import numpy as np
import json
import os
import sys


def cluster_positions(vals, gap=8):
    """Cluster sorted 1-D positions; merge values closer than `gap` px."""
    if not vals:
        return []
    vals = sorted(vals)
    clusters = [[vals[0]]]
    for v in vals[1:]:
        if v - clusters[-1][-1] <= gap:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [int(round(float(np.mean(c)))) for c in clusters]


def skew_angle(gray):
    """Estimate page tilt via projection profiles on a downscaled binary.

    Scans arrive tilted by up to a few degrees; a 1-2° tilt drifts a grid
    line several px across the long morphological kernels and breaks
    detection entirely (2025-11-16 page 1 was such a FAIL). The angle that
    maximizes row-projection variance makes the rows horizontal.
    """
    small = cv2.resize(gray, None, fx=0.25, fy=0.25)
    bw = cv2.adaptiveThreshold(
        small, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    h, w = bw.shape
    best_a, best_v = 0.0, -1.0
    for a in np.arange(-3.0, 3.01, 0.1):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        r = cv2.warpAffine(bw, m, (w, h))
        v = float(np.var(r.sum(axis=1)))
        if v > best_v:
            best_v, best_a = v, float(a)
    return best_a


def load_rotated(path):
    """Read a scan; portrait files hold landscape content -> rotate 90 CW.
    Also deskews, so every downstream tool sees the same geometry."""
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    a = skew_angle(gray)
    if abs(a) >= 0.15:
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        img = cv2.warpAffine(
            img, m, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
    return img


def detect(gray):
    """Detect long-line positions; returns dict with clustered H/V lines."""
    H, W = gray.shape
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (W // 25, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, H // 25))
    hlines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
    vlines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)

    result = {"W": W, "H": H}

    n, _, stats, _ = cv2.connectedComponentsWithStats(vlines)
    v_major = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if h >= 0.30 * H:
            v_major.append((x + w // 2, y, y + h))
    result["v_major_raw"] = sorted(v_major)
    result["v_clusters"] = cluster_positions([v[0] for v in v_major])

    n, _, stats, _ = cv2.connectedComponentsWithStats(hlines)
    h_major = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if w >= 0.40 * W:
            h_major.append((y + h // 2, x, x + w))
    result["h_major_raw"] = sorted(h_major)
    result["h_clusters"] = cluster_positions([h[0] for h in h_major])

    def spacings(pos):
        return [b - a for a, b in zip(pos, pos[1:])]

    result["v_spacings"] = spacings(result["v_clusters"])
    result["h_spacings"] = spacings(result["h_clusters"])
    result["_masks"] = (hlines, vlines)
    return result


def main():
    src, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)

    img = load_rotated(src)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    result = detect(gray)
    hlines, vlines = result.pop("_masks")
    result["image"] = os.path.basename(src)

    with open(os.path.join(outdir, "grid.json"), "w") as f:
        json.dump(result, f, indent=1)

    overlay = img.copy()
    overlay[hlines > 0] = (0, 200, 0)
    overlay[vlines > 0] = (0, 0, 255)
    for x in result["v_clusters"]:
        cv2.line(overlay, (x, 0), (x, H), (255, 0, 255), 2)
    for y in result["h_clusters"]:
        cv2.line(overlay, (0, y), (W, y), (255, 128, 0), 2)
    scale = 2000 / W
    small = cv2.resize(overlay, (2000, int(H * scale)))
    cv2.imwrite(os.path.join(outdir, "overlay.png"), small)

    print(f"v_clusters ({len(result['v_clusters'])}): {result['v_clusters']}")
    print(f"v_spacings: {result['v_spacings']}")
    print(f"h_clusters ({len(result['h_clusters'])}): {result['h_clusters']}")
    print(f"h_spacings: {result['h_spacings']}")


if __name__ == "__main__":
    main()
