#!/usr/bin/env python3
"""Score edge_score.py against a ground-truth transcription.

Compares the deterministic base-visit scorer to the human-read truth for
every cell of a game, in three layers:

  blank    -- pa:false cells should stay under the ink noise floor
  bases    -- furthest base reached (semantic; the scorer can only see
              what was DRAWN, and this scorer's hand skips the trace on
              some walks, so expected misses are reported, not hidden)
  scored   -- run scored vs center-mark ink, printed as the two
              distributions so a threshold can be picked from data

Usage: gt_score.py <image> <grid.json> <gt.json> <ref_r,ref_c>
"""
import cv2
import json
import re
import sys

import cell_diff
import edge_score
import grid_detect

REACHED_BASES = {None: 0, "1B": 1, "2B": 2, "3B": 3, "home": 3}


def main():
    src, gridpath, gtpath, ref = sys.argv[1:5]
    ref_rc = tuple(int(x) for x in ref.split(","))
    with open(gridpath) as f:
        grid = json.load(f)
    with open(gtpath) as f:
        gt = json.load(f)
    cols, rows = cell_diff.lattice(grid)
    gray = cv2.cvtColor(grid_detect.load_rotated(src), cv2.COLOR_BGR2GRAY)
    aligned, template, _ = cell_diff.build_aligned(gray, cols, rows, ref_rc)
    x_play = cell_diff.PAD + cell_diff.strip_width(grid["v_clusters"]) + 3

    def rc(tag):
        m = re.fullmatch(r"r(\d+)c(\d+)", tag)
        if not m:
            raise SystemExit(f"bad cell tag in ground truth: {tag}")
        return int(m.group(1)), int(m.group(2))

    blanks_ok = blanks_bad = 0
    exact = over = under = 0
    center_scored, center_not = [], []
    circle_cm = {(a, b): [] for a in (True, False) for b in (True, False)}
    misses = []
    for tag, truth in sorted(gt["cells"].items(), key=lambda kv: rc(kv[0])):
        ri, ci = rc(tag)
        if (ri, ci) not in aligned:
            print(f"{tag}: outside lattice, skipped")
            continue
        resid = cell_diff.residual(template, aligned[(ri, ci)])
        s, _ = edge_score.score_cell(resid, x_play)
        if not truth.get("pa"):
            if s["ink"] < 80:
                blanks_ok += 1
            else:
                blanks_bad += 1
                print(f"{tag}: BLANK but ink={s['ink']}")
            continue
        want = REACHED_BASES[truth.get("reached")]
        got = s["bases"]
        if got == want:
            exact += 1
        elif got > want:
            over += 1
            misses.append((tag, want, got, s))
        else:
            under += 1
            misses.append((tag, want, got, s))
        (center_scored if truth.get("scored") else center_not).append(
            (s["center_ink"], tag))
        circle_cm[(bool(truth.get("scored")), s["center_circle"])].append(tag)

    print(f"\nblank cells: {blanks_ok} ok / {blanks_bad} flagged")
    n_pa = exact + over + under
    print(f"bases (n={n_pa}): exact={exact} over={over} under={under}")
    for tag, want, got, s in misses:
        v = "".join(k[0] if s["visited"][k] else "." for k in edge_score.CORNERS)
        print(f"  {tag}: want={want} got={got} visited={v} "
              f"dist={s['corner_dist']} ink={s['ink']} "
              f"note={gt['cells'][tag].get('reach_via', '')!r}")
    for label, vals in (("scored", center_scored), ("not-scored", center_not)):
        vals.sort()
        line = " ".join(f"{v}:{t}" for v, t in vals)
        print(f"center_ink {label} (n={len(vals)}): {line}")
    for (truth_s, pred_c), tags in sorted(circle_cm.items()):
        kind = {(True, True): "hit", (True, False): "miss",
                (False, True): "false-alarm", (False, False): "reject"}
        print(f"circle {kind[(truth_s, pred_c)]:11s} n={len(tags):2d}"
              f"  {' '.join(tags) if kind[(truth_s, pred_c)] in ('miss', 'false-alarm') else ''}")


if __name__ == "__main__":
    main()
