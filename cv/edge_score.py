#!/usr/bin/env python3
"""Base-visit scorer: which bases did the batter reach?

The hand trace loops around the big diamond whose corners sit at the
cell-edge midpoints (bottom=home, right=1B, top=2B, left=3B). Two earlier
designs failed: sampling an ideal edge path (annotations sit on the path,
traces drift off it) and diagonal-orientation opening (real traces are
rounded squares whose local direction runs from horizontal to vertical).

What is invariant: the trace is a LONG connected stroke and it passes
close to the corner waypoints of every base reached, while annotations
(digits, parens, B/K) are compact and live in the quadrant interiors.
So: keep only long connected components of the residual and check the
distance from each corner waypoint to that stroke set.

Center mark (○/●/C) inside the diamond indicates a run scored.

Usage: edge_score.py <image> <grid.json> <outdir> <ref_r,ref_c> [r,c ...]
       With no r,c picks, scores every batter cell with ink above noise.
"""
import cv2
import numpy as np
import json
import os
import sys

import cell_diff
import grid_detect

CORNERS = ["home", "1B", "2B", "3B"]
INK_ON = 40        # residual value considered ink (pencil can be faint)
LONG_EXTENT = 65   # min bbox side of a component to count as a trace stroke
                   # (shortest real edge stroke ~70-85px; dilated digit
                   # chains like "(3)" stay under ~55px)
CORNER_R = 16      # a corner is "visited" if a long stroke comes this close
CORNER_INSET = 10  # corner waypoints sit this far inside the play area

# Long components that are NOT trace strokes, measured on the reference
# game (cv/ground-truth/2024-12-01_002.json diag, 2026-07-04): printed
# border/strip-separator lines survive the template diff when a cell
# aligns imperfectly, and out annotations (III + K) can merge into one
# big blob. Real traces measured dens 0.10-0.30 and band <= 0.59;
# residue lines band 0.88-1.00 (or sparse L-shapes band 0.54 dens 0.06);
# the merged-annotation blob dens 0.34. Margins are thin -- tuned on one
# game, revisit as more ground truth accumulates.
BAND_HW = 7         # half-width of the printed-line bands (px)
BAND_DROP = 0.6     # component mostly inside line bands = residue
BAND_SPARSE = 0.45  # ...or half in-band and nearly hollow = residue L
DENS_SPARSE = 0.10
DENS_DROP = 0.32    # bbox fill this dense = annotation blob, not stroke

CENTER_HOLE_R = 24    # scored-mark (○) search half-size around center
CENTER_HOLE_MIN = 12  # enclosed background px that make a closed loop


def corner_points(shape, x_play):
    """Base waypoints: play-area edge midpoints (bottom=home, right=1B, ...).

    x_play excludes the ball/strike count strip at the cell's left."""
    h, w = shape
    m = CORNER_INSET
    cx = (x_play + w) // 2
    return {
        "home": (cx, h - m),
        "1B": (w - m, h // 2),
        "2B": (cx, m),
        "3B": (x_play + m, h // 2),
    }


def score_cell(resid, x_play=0):
    """Base-visit detection.

    Hand traces are rounded squares more than diamonds — edge orientation
    is useless (a rounded corner runs from horizontal to vertical), so we
    do not look at directions at all. Instead: a base was reached when a
    LONG connected stroke passes near that base's corner waypoint.
    Annotations (digits, parens, B/K) are compact components that sit in
    the quadrant interiors, away from the waypoints.
    """
    h, w = resid.shape
    # diamond center = center of the PLAY area (past the count strip);
    # w//2 sat ~20px left of it and shifted the scored-mark probe
    cx, cy = (x_play + w) // 2, h // 2
    binr = ((resid >= INK_ON) * 255).astype(np.uint8)
    binr[:, :x_play] = 0  # count-strip marks are not base traces
    # bridge faint pencil gaps so one trace stays one component
    binr = cv2.dilate(binr, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binr)
    # neighbor cells' ink bleeds into the pad margin (e.g. the next
    # inning's trace runs along the shared boundary). A blanket margin
    # trim splits genuine loops, so instead drop components that live
    # MOSTLY in the margin — real traces pass through the interior.
    t = cell_diff.PAD + 2
    interior = np.zeros_like(binr, bool)
    interior[t:-t, t:-t] = True
    # printed-line bands: cell borders plus the count-strip separator.
    # Imperfect alignment leaves these lines in the residual and they are
    # long by construction, so components living in the bands are residue.
    band = np.zeros_like(binr, bool)
    pad = cell_diff.PAD
    for yb in (pad, h - pad):
        band[max(0, yb - BAND_HW):yb + BAND_HW, :] = True
    for xb in (pad, w - pad, x_play - 3):
        band[:, max(0, xb - BAND_HW):xb + BAND_HW] = True
    long_mask = np.zeros_like(binr)
    for i in range(1, n):
        x, y, ww, hh, area = (int(v) for v in stats[i])
        if max(ww, hh) < LONG_EXTENT:
            continue
        comp = labels == i
        if np.count_nonzero(comp & interior) < 0.5 * area:
            continue
        dens = area / (ww * hh)
        bfrac = np.count_nonzero(comp & band) / area
        if bfrac >= BAND_DROP or (bfrac >= BAND_SPARSE and dens < DENS_SPARSE):
            continue  # border/separator residue
        if dens >= DENS_DROP:
            continue  # merged annotation blob (e.g. III + K), not a stroke
        long_mask[comp] = 255
    pts = corner_points(resid.shape, x_play)
    ys, xs = np.nonzero(long_mask)
    visited, dist = {}, {}
    for name in CORNERS:
        px, py = pts[name]
        if len(xs) == 0:
            visited[name], dist[name] = False, None
            continue
        d = float(np.min(np.hypot(xs - px, ys - py)))
        dist[name] = round(d, 1)
        visited[name] = d <= CORNER_R
    # bases reached = consecutive visits beyond home
    bases = 0
    for name in ["1B", "2B", "3B"]:
        if not visited[name]:
            break
        bases += 1
    scored = bases == 3 and visited["home"]
    ctr = resid[cy - 14 : cy + 14, cx - 14 : cx + 14]
    center_ink = int(ctr[ctr > 40].sum() / 255)
    total_ink = int(resid[resid > 40].sum() / 255)
    # scored mark: the run circle (○) is a CLOSED curve at the diamond
    # center, so it encloses background; out numerals (I/II/III, K) are
    # open strokes and enclose nothing. center_ink alone cannot separate
    # the two (measured overlap 21-56 vs 0-92 on the reference game).
    half = CENTER_HOLE_R
    patch = (resid[cy - half:cy + half, cx - half:cx + half] >= INK_ON)
    inv = (~patch).astype(np.uint8)
    nn, _, st2, _ = cv2.connectedComponentsWithStats(inv)
    ph, pw = inv.shape
    hole_px = 0
    for j in range(1, nn):
        bx, by, bw, bh, barea = (int(v) for v in st2[j])
        if bx > 0 and by > 0 and bx + bw < pw and by + bh < ph:
            hole_px += barea
    return {
        "visited": visited,
        "corner_dist": dist,
        "bases": bases,
        "scored_loop": scored,
        "center_ink": center_ink,
        "center_circle": hole_px >= CENTER_HOLE_MIN,
        "center_hole_px": hole_px,
        "ink": total_ink,
    }, long_mask


def visualize(resid, score, long_mask, outpath, x_play=0):
    vis = cv2.cvtColor(255 - resid, cv2.COLOR_GRAY2BGR)
    vis[long_mask > 0] = (200, 120, 0)
    pts = corner_points(resid.shape, x_play)
    for name in CORNERS:
        color = (0, 180, 0) if score["visited"][name] else (0, 0, 230)
        cv2.circle(vis, pts[name], CORNER_R, color, 1)
    vis = cv2.resize(vis, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(outpath, vis)


def main():
    src, gridpath, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    ref_rc = tuple(int(x) for x in sys.argv[4].split(","))
    picks = [tuple(int(x) for x in a.split(",")) for a in sys.argv[5:]]
    os.makedirs(outdir, exist_ok=True)

    with open(gridpath) as f:
        grid = json.load(f)
    cols, rows = cell_diff.lattice(grid)

    gray = cv2.cvtColor(grid_detect.load_rotated(src), cv2.COLOR_BGR2GRAY)
    aligned, template, _ = cell_diff.build_aligned(gray, cols, rows, ref_rc)
    # play area starts after PAD + count strip + its separator line
    x_play = cell_diff.PAD + cell_diff.strip_width(grid["v_clusters"]) + 3

    auto_scan = not picks
    if not picks:
        picks = sorted(aligned.keys())
    report = {}
    for ri, ci in picks:
        resid = cell_diff.residual(template, aligned[(ri, ci)])
        s, long_mask = score_cell(resid, x_play)
        # blank filter only in auto scan; explicitly picked cells always report
        if auto_scan and s["ink"] < 80:  # blank noise floor (measured: ~20)
            continue
        report[f"r{ri}c{ci}"] = s
        visualize(resid, s, long_mask,
                  os.path.join(outdir, f"edges_r{ri}c{ci}.png"), x_play)
        v = "".join(name[0] if s["visited"][name] else "." for name in CORNERS)
        print(
            f"r{ri}c{ci}: visited={v} dist={s['corner_dist']} "
            f"bases={s['bases']} loop={s['scored_loop']} "
            f"circle={s['center_hole_px']} center={s['center_ink']} "
            f"ink={s['ink']}"
        )
    with open(os.path.join(outdir, "edge_scores.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
