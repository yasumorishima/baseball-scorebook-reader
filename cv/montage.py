#!/usr/bin/env python3
"""Residual montage for ground-truth reading.

Lays out the handwriting residual of every batter cell in batting-order x
inning-column grid, labeled rXcY, so one image carries a whole game for
visual ground-truth annotation (in-session LLM vision or human).

Usage: montage.py <image> <grid.json> <out.png> <ref_r,ref_c> [c0-c6]
"""
import cv2
import numpy as np
import json
import sys

import cell_diff
import grid_detect


def main():
    src, gridpath, outpath = sys.argv[1], sys.argv[2], sys.argv[3]
    ref_rc = tuple(int(x) for x in sys.argv[4].split(","))
    col_lo, col_hi = 0, 12
    if len(sys.argv) > 5:
        col_lo, col_hi = (int(x.lstrip("c")) for x in sys.argv[5].split("-"))

    with open(gridpath) as f:
        grid = json.load(f)
    cols, rows = cell_diff.lattice(grid)
    col_lo = max(0, col_lo)
    col_hi = min(col_hi, len(cols) - 1)
    gray = cv2.cvtColor(grid_detect.load_rotated(src), cv2.COLOR_BGR2GRAY)
    aligned, template, _ = cell_diff.build_aligned(gray, cols, rows, ref_rc)

    ch, cw = template.shape
    scale = 2
    label_h = 22
    tile_w, tile_h = cw * scale, ch * scale + label_h
    n_r = len(rows) - 1
    n_c = col_hi - col_lo + 1
    canvas = np.full((tile_h * n_r, tile_w * n_c, 3), 255, np.uint8)
    for ri in range(1, len(rows)):
        for ci in range(col_lo, col_hi + 1):
            resid = cell_diff.residual(template, aligned[(ri, ci)])
            vis = cv2.cvtColor(255 - resid, cv2.COLOR_GRAY2BGR)
            vis = cv2.resize(vis, (cw * scale, ch * scale),
                             interpolation=cv2.INTER_CUBIC)
            y = (ri - 1) * tile_h
            x = (ci - col_lo) * tile_w
            cv2.putText(canvas, f"r{ri}c{ci}", (x + 4, y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 0, 0), 1)
            canvas[y + label_h : y + tile_h, x : x + tile_w] = vis
            cv2.rectangle(canvas, (x, y), (x + tile_w - 1, y + tile_h - 1),
                          (200, 200, 200), 1)
    cv2.imwrite(outpath, canvas)
    print(f"montage {canvas.shape[1]}x{canvas.shape[0]} -> {outpath}")


if __name__ == "__main__":
    main()
