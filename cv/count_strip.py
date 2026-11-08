#!/usr/bin/env python3
"""Ball/strike count-strip reader.

Each inning column carries a narrow (~24px) strip at its left where the
recorder marks the pitch sequence (circles, dots, short tallies, small
triangles). This layer counts the residual-ink marks in the strip and
reports shape features per mark; the semantic mapping (ball vs strike vs
foul) is decided later by the fusion layer against ground truth.

Usage: count_strip.py <image> <grid.json> <outdir> <ref_r,ref_c> [r,c ...]
       With no picks, reports every batter cell whose strip has marks.
"""
import cv2
import numpy as np
import json
import os
import sys

import cell_diff
import grid_detect

INK_ON = 40
MIN_AREA = 12      # smallest real pen mark after binarization
LINE_ASPECT = 2.2  # bbox aspect ratio above which a mark reads as a tally


def strip_marks(resid, strip_w):
    """Connected-component marks inside the count strip."""
    h, w = resid.shape
    x0 = cell_diff.PAD + 2                # inside the column's left rule
    x1 = cell_diff.PAD + strip_w          # exclude the strip separator rule
                                          # (its diff residue would read as
                                          # a phantom vertical "tally")
    y0, y1 = cell_diff.PAD + 2, h - cell_diff.PAD - 2
    strip = resid[y0:y1, x0:x1]
    binr = ((strip >= INK_ON) * 255).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binr)
    marks = []
    for i in range(1, n):
        x, y, ww, hh, area = (int(v) for v in stats[i])
        if area < MIN_AREA:
            continue
        aspect = max(ww, hh) / max(1, min(ww, hh))
        fill = area / max(1, ww * hh)
        if aspect >= LINE_ASPECT:
            shape = "tally"
        elif fill >= 0.55:
            shape = "dot"
        else:
            shape = "ring"
        marks.append(
            {
                "y_frac": round((y + hh / 2) / strip.shape[0], 2),
                "shape": shape,
                "area": area,
                "wh": [ww, hh],
            }
        )
    marks.sort(key=lambda m: m["y_frac"])
    return marks


def main():
    src, gridpath, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    ref_rc = tuple(int(x) for x in sys.argv[4].split(","))
    picks = [tuple(int(x) for x in a.split(",")) for a in sys.argv[5:]]
    os.makedirs(outdir, exist_ok=True)

    with open(gridpath) as f:
        grid = json.load(f)
    cols, rows = cell_diff.lattice(grid)
    strip_w = cell_diff.strip_width(grid["v_clusters"])

    gray = cv2.cvtColor(grid_detect.load_rotated(src), cv2.COLOR_BGR2GRAY)
    aligned, template, _ = cell_diff.build_aligned(gray, cols, rows, ref_rc)

    auto_scan = not picks
    if not picks:
        picks = sorted(aligned.keys())
    report = {}
    for ri, ci in picks:
        resid = cell_diff.residual(template, aligned[(ri, ci)])
        marks = strip_marks(resid, strip_w)
        if auto_scan and not marks:
            continue
        report[f"r{ri}c{ci}"] = marks
        counts = {}
        for m in marks:
            counts[m["shape"]] = counts.get(m["shape"], 0) + 1
        print(f"r{ri}c{ci}: n={len(marks)} {counts} "
              f"y={[m['y_frac'] for m in marks]}")
    with open(os.path.join(outdir, "strip_marks.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
