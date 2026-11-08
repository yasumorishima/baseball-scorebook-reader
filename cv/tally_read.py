#!/usr/bin/env python3
"""Right-edge stats-strip tally reader, diffed against the ground truth.

The scorebook's right edge carries 16 per-player total columns (PA,
AB, runs, 1B/2B/3B/HR, total bases, RBI, SB, CS, sac, SF, BB+HBP, K,
LOB) tallied with SEI strokes (ICHI=1, T=2, SHITA=3, SEI=5), pencil
over green print. This tool reads the six derivable columns per batter
row and diffs them against the same stats derived from the transcribed
GT -- a solver-independent cross-check in the spirit of the header
line scores (which settled the 2025-10-19 score), automated.

What the 25-sheet corpus actually offers (measured 2026-07-12): only
THREE sheets (2024-12-01 own, 2025-11-16 opp, 2025-09-14 own) carry a
filled strip; 21 strips are blank (most recorders never fill the
totals) and 2026-05-24_01 uses a lattice this reader does not resolve
(13 separators). The bottom totals block is just as bare (the one
sheet that uses it fills pitch counts only). So sheet-internal
redundancy is thin in this corpus, and the header line score remains
the only broadly available cross-signal. On the three filled sheets
the reader agrees with GT-derived stats at SB 93% / 1B 90% / K 87% /
BB 83% / PA 73% / AB 37% exact -- the AB column is dominated by the
recorders' own live-aggregation slips (visible strokes genuinely
number fewer than the plate appearances), which is exactly what the
diff is meant to surface, plus a known R-column omission habit (one
recorder tallies RBI but never runs).

Recognition: per-cell adaptive pencil mask (gray < cell median - 25,
green-tinted print suppressed by G-(R+B)/2 < 8, 2x2 opening), stroke
count = connected components (>= 6 px) of horizontal + vertical
morphological openings (kernel 12). Measured on the hand-checked game
1 calibration set: 44/59 exact, MAE 0.31. Probed and REJECTED:
diagonal openings on the residual (never fires -- the slanted third
tick of a cursive SHITA is too short), skeleton-segment counting
(41-42/59), dark-core dash suppression (no effect: the axis openings
already reject the printed dash guides), skeleton-length / unit-length
regression (1.05-1.8x spread).

Usage: tally_read.py <games-manifest.json> [game-id ...]
"""
import collections
import json
import os
import re
import sys

import cv2
import numpy as np

import cell_diff
import grid_detect
from stats_crop import stats_span

COLIDX = {"PA": 0, "AB": 1, "B1": 3, "SB": 9, "BB": 13, "K": 14}
PADC = 4          # stay off the printed separators
KLEN = 12         # stroke opening length
MIN_STROKE = 6    # component px floor
EMPTY_FRAC = 0.4  # strip counted empty when PA strokes < this * GT PA


def derive_rows(gt):
    """Per-row PA/AB/BB/K/SB/1B from a GT file. Same classification
    rules as the 2025 site crosscheck (reports/2025_gt_site_crosscheck):
    the sheet's BB column combines walks and HBP, and both fold into
    the BB counter here, so AB = PA - BB covers HBP too (sac flies are
    judged manually upstream and are rare); ambiguous reach
    descriptions poison only the 1B cell, not the row."""
    rows = {}
    for key, v in gt["cells"].items():
        m = re.match(r"r(\d+)c(\d+)", key)
        if not m or not v.get("pa"):
            continue
        r = int(m.group(1))
        st = rows.setdefault(r, collections.Counter())
        st["PA"] += 1
        rv = v.get("reach_via") or ""
        low = rv.lower()
        out = v.get("out") or {}
        play = str(out.get("play", ""))
        amb = bool(rv) and any(w in low for w in
                               (" or ", "unclear", "unreadable",
                                "unresolved", "unmarked", "trails off"))
        if rv.startswith(("B", "D.B", "DB", "D-B")):
            st["BB"] += 1
        elif amb:
            st["AMB"] += 1
        elif rv and (rv.startswith("/") or "single" in low
                     or "infield hit" in low or "double" in low
                     or "triple" in low or "home run" in low
                     or rv.strip() == "HR"):
            if not any(w in low for w in ("double", "triple", "home run")) \
                    and rv.strip() != "HR":
                st["B1"] += 1
        if out.get("as") == "batter" and play.startswith("K"):
            st["K"] += 1
        if rv.startswith(("K", "k")):   # uncaught third strike reach
            st["K"] += 1
        st["SB"] += sum(1 for x in (v.get("advances") or {}).values()
                        if x == "S")
    for st in rows.values():
        st["AB"] = st["PA"] - st["BB"]
    return rows


def pencil_mask(gray, greenness, y0, y1, sx0, sx1):
    cell = gray[y0:y1, sx0:sx1]
    med = int(np.median(cell))
    m = ((cell < med - 25) &
         (greenness[y0:y1, sx0:sx1] < 8)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))


def _ncomp(m):
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    return sum(1 for i in range(1, n) if st[i][4] >= MIN_STROKE)


def stroke_count(m):
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (KLEN, 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, KLEN))
    return (_ncomp(cv2.morphologyEx(m, cv2.MORPH_OPEN, kh)) +
            _ncomp(cv2.morphologyEx(m, cv2.MORPH_OPEN, kv)))


def read_game(g):
    """(status, est, truth) -- status in filled/empty/no-lattice."""
    with open(g["gt"]) as f:
        gt = json.load(f)
    truth = derive_rows(gt)
    with open(g["grid"]) as f:
        grid = json.load(f)
    try:
        x0, x1 = stats_span(grid["v_clusters"])
    except SystemExit:
        return "no-lattice", {}, truth
    seps = [s for s in grid["v_clusters"] if x0 <= s <= x1]
    if len(seps) < 16:   # need PA..K = indices 0..14
        return "no-lattice", {}, truth
    _, rows = cell_diff.lattice(grid)
    img = grid_detect.load_rotated(g["image"])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
    b, gr, r = (img[..., c].astype(np.int16) for c in (0, 1, 2))
    greenness = gr - (r + b) // 2

    est = {}
    for name, ci in COLIDX.items():
        for ri in truth:
            if ri >= len(rows):
                continue
            y0, y1 = rows[ri]
            m = pencil_mask(gray, greenness, y0 + PADC, y1 - PADC,
                            seps[ci] + PADC, seps[ci + 1] - PADC)
            est[(name, ri)] = stroke_count(m)
    pa_sum = sum(v for (n, _), v in est.items() if n == "PA")
    pa_truth = sum(st["PA"] for st in truth.values())
    if pa_sum < EMPTY_FRAC * pa_truth:
        return "empty", est, truth
    return "filled", est, truth


def main():
    with open(sys.argv[1]) as f:
        games = json.load(f)
    only = set(sys.argv[2:])
    agree = collections.Counter()
    tot = collections.Counter()
    diffs = []
    status_by = {}
    for g in games:
        if only and g["id"] not in only:
            continue
        status, est, truth = read_game(g)
        status_by[g["id"]] = status
        if status != "filled":
            continue
        for name in COLIDX:
            for ri, st in truth.items():
                if (name, ri) not in est:
                    continue
                if name == "B1" and st["AMB"]:
                    continue
                tot[name] += 1
                if est[(name, ri)] == st[name]:
                    agree[name] += 1
                else:
                    diffs.append((g["id"], name, ri, st[name],
                                  est[(name, ri)]))
    for s in ("filled", "empty", "no-lattice"):
        ids = sorted(k for k, v in status_by.items() if v == s)
        print(f"{s}: {len(ids)}  {' '.join(ids)}")
    print("\nagreement (exact):")
    for name in COLIDX:
        if tot[name]:
            print(f"  {name:3s} {agree[name]}/{tot[name]} "
                  f"= {agree[name] / tot[name]:.0%}")
    print(f"\nsheet-vs-GT differences ({len(diffs)}) -- estimator error "
          f"or recorder aggregation slip, audit candidates:")
    for d in diffs:
        print("  %s %s r%d gt=%d sheet~%d" % d)


if __name__ == "__main__":
    main()
