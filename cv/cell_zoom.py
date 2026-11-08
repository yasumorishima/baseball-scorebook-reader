#!/usr/bin/env python3
"""Per-cell zoom pairs (residual | color) for ground-truth annotation.

The montage resolves most cells, but crowded annotations (ell marks,
error notations, advancement numbers) need more pixels. This renders
the requested cells one file each at high scale, residual on the left
(print removed) and the raw color crop on the right (graphite-vs-print
color cue preserved).

Usage: cell_zoom.py <image> <grid.json> <ref_r,ref_c> <outdir> r1c0 r5c3 ...
       CELL_ZOOM_MARGIN=45 widens the color-crop context (annotations
       sometimes spill past the cell border into the row below).
"""
import cv2
import numpy as np
import json
import os
import re
import sys

import cell_diff
import grid_detect

SCALE = 4
MARGIN = int(os.environ.get("CELL_ZOOM_MARGIN", "10"))   # color-crop context (px)


def main():
    src, gridpath, ref, outdir = sys.argv[1:5]
    tags = sys.argv[5:]
    ref_rc = tuple(int(x) for x in ref.split(","))
    with open(gridpath) as f:
        grid = json.load(f)
    cols, rows = cell_diff.lattice(grid)
    img = grid_detect.load_rotated(src)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aligned, template, _ = cell_diff.build_aligned(gray, cols, rows, ref_rc)

    os.makedirs(outdir, exist_ok=True)
    for tag in tags:
        m = re.fullmatch(r"r(\d+)c(\d+)", tag)
        if not m:
            raise SystemExit(f"bad cell tag: {tag}")
        ri, ci = int(m.group(1)), int(m.group(2))
        if not (1 <= ri < len(rows) and 0 <= ci < len(cols)):
            raise SystemExit(f"{tag} outside lattice "
                             f"(rows 1..{len(rows) - 1}, cols 0..{len(cols) - 1})")
        resid = cell_diff.residual(template, aligned[(ri, ci)])
        left = cv2.cvtColor(255 - resid, cv2.COLOR_GRAY2BGR)
        y0, y1 = rows[ri]
        x0, x1 = cols[ci]
        crop = img[max(0, y0 - MARGIN):y1 + MARGIN,
                   max(0, x0 - MARGIN):x1 + MARGIN]
        h = max(left.shape[0], crop.shape[0])
        pair = np.full((h, left.shape[1] + crop.shape[1] + 8, 3), 255,
                       np.uint8)
        pair[:left.shape[0], :left.shape[1]] = left
        pair[:crop.shape[0], left.shape[1] + 8:] = crop
        pair = cv2.resize(pair, None, fx=SCALE, fy=SCALE,
                          interpolation=cv2.INTER_CUBIC)
        cv2.putText(pair, tag, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (180, 0, 0), 2)
        cv2.imwrite(os.path.join(outdir, f"{tag}.png"), pair)
    print(f"{len(tags)} cell zooms -> {outdir}")


if __name__ == "__main__":
    main()
