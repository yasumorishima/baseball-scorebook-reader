#!/usr/bin/env python3
"""Template differencing: isolate handwritten ink in play-grid cells.

Every play-grid cell is aligned to a reference (blank) cell via phase
correlation (scan distortion is real: up to ~12px vertical drift across a
page), then the per-pixel median over all aligned cells reconstructs the
blank printed form; subtraction cancels the print and leaves handwriting.

Row 0 of the detected lattice is the inning-number header row; batters
start at row 1 (batting-order slot 1 = r1, slot 2 = r2, ...).

Usage: cell_diff.py <image> <grid.json> <outdir> <ref_r,ref_c> [r,c ...]
"""
import cv2
import numpy as np
import json
import os
import sys

import grid_detect

PAD = 6  # crop margin so alignment shifts don't pull in neighbors


def inning_col_bounds(v_clusters):
    """Inning columns show up as (narrow ~24px count strip, ~132px cell)."""
    starts = []
    for i in range(len(v_clusters) - 2):
        a, b, c = v_clusters[i : i + 3]
        if 18 <= b - a <= 32 and 120 <= c - b <= 145:
            starts.append(a)
    if len(starts) < 2:
        raise SystemExit(f"inning columns not recognized (starts={starts})")
    width = int(round(float(np.median(np.diff(starts)))))
    # impute starts the strict pattern missed (e.g. a faded count-strip
    # separator): the lattice is periodic, so a gap of ~k*width between
    # consecutive detected starts hides k-1 columns at width intervals
    full = [starts[0]]
    for s in starts[1:]:
        gap = s - full[-1]
        k = int(round(gap / width))
        if k >= 2 and abs(gap - k * width) <= 0.1 * width * k:
            base = full[-1]
            step = gap / k
            for j in range(1, k):  # k-1 hidden columns, evenly spaced
                full.append(int(round(base + j * step)))
        full.append(s)
    return [(s, s + width) for s in full]


def batter_row_bounds(h_clusters):
    """Longest consecutive run of ~133px-spaced horizontal lines."""
    rows = []
    for a, b in zip(h_clusters, h_clusters[1:]):
        if 120 <= b - a <= 145:
            rows.append((a, b))
    if not rows:
        raise SystemExit("no batter rows recognized")
    runs, cur = [], [rows[0]]
    for r in rows[1:]:
        if r[0] == cur[-1][1]:
            cur.append(r)
        else:
            runs.append(cur)
            cur = [r]
    runs.append(cur)
    return max(runs, key=len)


def lattice(grid):
    return inning_col_bounds(grid["v_clusters"]), batter_row_bounds(
        grid["h_clusters"]
    )


def strip_width(v_clusters):
    """Width of the narrow ball/strike count strip at each column's left."""
    widths = []
    for a, b in zip(v_clusters, v_clusters[1:]):
        if 18 <= b - a <= 32:
            widths.append(b - a)
    if not widths:
        raise SystemExit("count strips not recognized")
    return int(round(float(np.median(widths))))


def build_aligned(gray, cols, rows, ref_rc):
    """Align all cells below the header row; return (aligned, template).

    aligned maps (row, col) -> uint8 cell of shape (ch+2*PAD, cw+2*PAD).
    """
    cw = min(c1 - c0 for c0, c1 in cols)
    ch = min(r1 - r0 for r0, r1 in rows)
    # pad the page once so edge cells crop in-bounds with identical shapes
    # (a raw y0-PAD slice would go negative at the page edge and silently
    # crop the wrong region; y0+ch+PAD past the edge would change shape)
    gpad = cv2.copyMakeBorder(gray, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)

    def crop(ri, ci):
        y0, x0 = rows[ri][0], cols[ci][0]
        return gpad[y0 : y0 + ch + 2 * PAD, x0 : x0 + cw + 2 * PAD]

    ref = crop(*ref_rc).astype(np.float32)

    def align(cell):
        c = cell.astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(ref, c)
        m = np.float32([[1, 0, -dx], [0, 1, -dy]])
        return cv2.warpAffine(
            cell, m, (cell.shape[1], cell.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        ), (dx, dy)

    aligned, shifts = {}, []
    for ri in range(1, len(rows)):  # skip header row
        for ci in range(len(cols)):
            a, (dx, dy) = align(crop(ri, ci))
            aligned[(ri, ci)] = a
            shifts.append((abs(dx), abs(dy)))
    if not aligned:
        raise SystemExit("no data rows below the header row")
    template = np.median(
        np.stack([a for a in aligned.values()]), axis=0
    ).astype(np.uint8)
    mx = round(max(s[0] for s in shifts), 1)
    my = round(max(s[1] for s in shifts), 1)
    return aligned, template, (mx, my)


def residual(template, cell):
    """Handwritten-ink map (uint8, ink -> bright)."""
    return cv2.subtract(template, cell)


def main():
    src, gridpath, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    ref_rc = tuple(int(x) for x in sys.argv[4].split(","))
    picks = [tuple(int(x) for x in a.split(",")) for a in sys.argv[5:]]
    os.makedirs(outdir, exist_ok=True)

    with open(gridpath) as f:
        grid = json.load(f)
    cols, rows = lattice(grid)

    gray = cv2.cvtColor(grid_detect.load_rotated(src), cv2.COLOR_BGR2GRAY)

    aligned, template, (mx, my) = build_aligned(gray, cols, rows, ref_rc)
    cv2.imwrite(os.path.join(outdir, "template.png"),
                cv2.resize(template, None, fx=3, fy=3))
    print(f"template from {len(aligned)} aligned cells, max |shift| x={mx} y={my}")

    for ri, ci in picks:
        cell = aligned[(ri, ci)]
        resid = residual(template, cell)
        resid_vis = 255 - resid
        h = cell.shape[0]
        panel = np.hstack([cell, np.full((h, 4), 128, np.uint8), resid_vis])
        panel = cv2.resize(panel, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(outdir, f"cell_r{ri}c{ci}.png"), panel)
        ink = int(resid[resid > 40].sum() / 255)
        print(f"r{ri}c{ci}: ink_score={ink}")


if __name__ == "__main__":
    main()
