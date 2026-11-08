#!/usr/bin/env python3
"""Right-edge stats strip montage for ground-truth cross-checking.

The scorebook's right edge carries per-player total columns (at-bats,
hits, runs, ...) as handwritten digits over green print. Template
differencing is unnecessary here — a plain color crop reads fine — but
the region must be located from the lattice, not hard-coded: it is the
trailing run of narrow (~40px) column separators in v_clusters.

Emits one labeled montage: header row (green printed column titles) on
top, then every batter row's stats strip, each tagged rX to match the
residual montage / ground-truth indices.

Usage: stats_crop.py <image> <grid.json> <out.png> [scale]
       (if <out> ends with "/", writes per-row files hdr.png, r1.png, ...
        into that directory instead of one montage)
"""
import cv2
import numpy as np
import json
import os
import sys

import cell_diff
import grid_detect

NARROW_MIN, NARROW_MAX = 30, 55   # stats separator spacing (px)
MIN_RUN = 5                       # need this many consecutive narrow gaps


def stats_span(v_clusters):
    """(x0, x1) of the trailing run of narrow gaps = stats region.

    Both edges come from the run itself — the last v_cluster on the page
    may be an outer border past the stats columns, which would widen the
    crop and shift the burned-in sub-column labels.
    """
    run_start = None
    run_len = 0
    best = None
    prev = None
    for a, b in zip(v_clusters, v_clusters[1:]):
        if NARROW_MIN <= b - a <= NARROW_MAX:
            if run_start is None:
                run_start = a
            run_len += 1
            prev = b
        else:
            if run_len >= MIN_RUN:
                best = (run_start, prev)
            run_start, run_len = None, 0
    if run_len >= MIN_RUN:
        best = (run_start, prev)
    if best is None:
        raise SystemExit("no trailing narrow-gap run found — stats "
                         "columns not detected in this lattice")
    return best


def main():
    src, gridpath, outpath = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(gridpath) as f:
        grid = json.load(f)
    v = grid["v_clusters"]
    x0, x1 = stats_span(v)
    # use the SAME lattice rows as montage.py so the rX tags here index
    # the same physical rows as the residual montage / ground truth
    # (row 0 = header row, batters start at r1)
    _, rows = cell_diff.lattice(grid)
    img = grid_detect.load_rotated(src)

    scale = max(1, int(sys.argv[4])) if len(sys.argv) > 4 else 2
    split = outpath.endswith("/") or outpath.endswith(os.sep)
    label_w = 60
    strip_w = (x1 - x0) * scale + label_w
    tiles = []
    entries = [("hdr" if i == 0 else f"r{i}", y0, y1)
               for i, (y0, y1) in enumerate(rows)]
    seps = [s for s in v if x0 <= s <= x1]
    print(f"stats region x0={x0} x1={x1} rows={len(rows)}")
    for tag, y0, y1 in entries:
        crop = img[y0:y1, x0:x1]
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
        # burn in the detected separators + sub-column indices so a
        # human/LLM reading the crop cannot mis-assign drifting marks
        if tag != "hdr":
            for k, s in enumerate(seps):
                sx = (s - x0) * scale
                cv2.line(crop, (sx, 0), (sx, crop.shape[0]), (0, 0, 220), 1)
                if k < len(seps) - 1:
                    cv2.putText(crop, str(k + 1), (sx + 4, 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 0, 220), 1)
        if split:
            os.makedirs(outpath, exist_ok=True)
            cv2.imwrite(os.path.join(outpath, f"{tag}.png"), crop)
            continue
        tile = np.full((crop.shape[0], strip_w, 3), 255, np.uint8)
        tile[:, label_w:label_w + crop.shape[1]] = crop
        cv2.putText(tile, tag, (4, crop.shape[0] // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 0, 0), 2)
        cv2.line(tile, (0, tile.shape[0] - 1),
                 (strip_w, tile.shape[0] - 1), (200, 200, 200), 1)
        tiles.append(tile)
    if split:
        print(f"per-row crops -> {outpath}")
        return
    canvas = np.vstack(tiles)
    cv2.imwrite(outpath, canvas)
    print(f"montage {canvas.shape[1]}x{canvas.shape[0]} -> {outpath}")


if __name__ == "__main__":
    main()
