#!/usr/bin/env python3
"""Detect hand-renumbered inning headers via color separation.

The printed form (inning numbers included) is GREEN ink; handwriting is
neutral-gray pencil. Template differencing cannot work on the header row
(each cell's print differs), but color can: classify dark pixels into
green-print vs neutral-pencil and flag header cells whose pencil mass
says the recorder overwrote the inning number (which happens when a big
inning overflows into the next printed column).

Emits per-column pencil/print pixel counts plus an upscaled color crop
of every flagged cell for human (or LLM-vision) digit reading.

Usage: header_scan.py <image> <grid.json> <outdir>
"""
import cv2
import numpy as np
import json
import os
import sys

import cell_diff
import grid_detect

INK_LUMA = 190      # darker than this counts as ink
GREEN_MIN = 12      # G - (R+B)/2 above this = printed green
NEUTRAL_MAX = 14    # max channel spread for pencil (gray) ink
PENCIL_FLAG = 100   # eroded pencil pixels above this = overwritten


def main():
    src, gridpath, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(outdir, exist_ok=True)

    with open(gridpath) as f:
        grid = json.load(f)
    cols, rows = cell_diff.lattice(grid)
    img = grid_detect.load_rotated(src)

    y0, y1 = rows[0]  # lattice row 0 = the inning-number header row
    # sanity: header cells carry a bold green printed digit (~1000-2000 px);
    # if the lattice run missed the header row, rows[0] is a batter cell
    # (light dashed print) and pencil-heavy cells would silently flip to
    # false OVERWRITTEN — fail loudly instead
    report = {}
    for ci, (x0, x1) in enumerate(cols):
        cell = img[y0:y1, x0:x1].astype(np.int32)
        b, g, r = cell[..., 0], cell[..., 1], cell[..., 2]
        luma = (b + g + r) // 3
        ink = luma < INK_LUMA
        greenness = g - (r + b) // 2
        spread = np.maximum(np.maximum(np.abs(r - g), np.abs(g - b)),
                            np.abs(r - b))
        print_px = int(np.count_nonzero(ink & (greenness > GREEN_MIN)))
        # the green print's anti-aliased rim decodes as neutral 1px noise;
        # a 2x2 erosion kills it while 2-4px pencil strokes survive
        pencil = ((ink & (spread <= NEUTRAL_MAX)) * 255).astype(np.uint8)
        pencil = cv2.erode(pencil, np.ones((2, 2), np.uint8))
        pencil_px = int(np.count_nonzero(pencil))
        overwritten = pencil_px >= PENCIL_FLAG
        report[f"c{ci}"] = {
            "print_px": print_px,
            "pencil_px": pencil_px,
            "overwritten": overwritten,
        }
        if overwritten:
            crop = cv2.resize(
                img[y0:y1, x0:x1], None, fx=3, fy=3,
                interpolation=cv2.INTER_CUBIC,
            )
            cv2.imwrite(os.path.join(outdir, f"header_c{ci}.png"), crop)
        print(f"c{ci}: print={print_px} pencil={pencil_px}"
              f"{' OVERWRITTEN' if overwritten else ''}")
    med_print = sorted(r["print_px"] for r in report.values())[len(report) // 2]
    if med_print < 800:
        raise SystemExit(
            f"row 0 does not look like the header row (median print_px="
            f"{med_print} < 800) — lattice may have missed the header"
        )
    with open(os.path.join(outdir, "header_scan.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
