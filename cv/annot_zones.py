#!/usr/bin/env python3
"""Annotation zone decomposition: where does each pencil cluster sit?

The notation puts meaning in POSITION before shape: corner paren
numbers mark advances (upper-right=2B, upper-left=3B, lower-left=home),
the lower-right quadrant holds how first base was reached (B, DB, /9,
5-4E) or the fielding play of an out, and the diamond center holds the
out numeral or the run circle. So before any character recognition, the
solver needs "which zones have writing" -- this tool extracts compact
annotation clusters (everything that is not a trace stroke, printed-line
residue, or count-strip mark) and bins them by zone.

With a ground-truth JSON it scores corner-cluster presence against the
transcribed advances -- the key solver input.

MIN_INK / CENTER_R / the 0.6 bleed cut are tuned on the one transcribed
game (29/30 numbered advances found); revisit as ground truth grows.

Usage: annot_zones.py <image> <grid.json> <ref_r,ref_c> <out.json> [gt.json]
"""
import cv2
import numpy as np
import json
import os
import re
import sys

import cell_diff
import edge_score
import grid_detect

CLUSTER_GAP = 1   # dilation radius when grouping ink into clusters.
                  # Anything wider bridges the printed dashed-diamond
                  # leftovers (dashes sit 4-6px apart), which chains every
                  # annotation touching the ring into one mega cluster
                  # whose centroid reads "center". At 1 the dashes stay
                  # isolated specks below MIN_INK and characters split
                  # into pieces at worst -- pieces still land in the
                  # right zone, which is what the solver consumes.
MIN_INK = 40      # ignore specks below this many ink px
# Weld surgery v2 (2026-07-27). The transit path was chosen as the
# farthest PAIR of border skeleton pixels without requiring the two to
# be connected, so a zone holding both the trace and a glyph that also
# reaches the border picks one endpoint on each and the BFS can never
# arrive -- the whole surgery then aborts. Measured on the eight zones
# that still yield no crop: two die that way and two more die because
# only a single skeleton pixel touches the border at all. Restricting
# the pair to one component is a strict superset of the old rule (when
# the old rule succeeds the pair is already connected, so the same path
# comes out); the border fallback is the genuinely new part, and the
# downstream glyph-vs-arc gate still guards what it produces.
WELD2 = os.environ.get("CORNER_WELD2", "1") == "1"
# Zone membership is decided by a single centroid, so a mark written
# near a quadrant boundary belongs to whichever side its centre of mass
# lands on. Measured 2026-07-27: only 3 of 347 corners have a cluster
# filed elsewhere that holds more ink inside their box, so this is a
# small effect -- but small is not zero and it is cheap to test.
ZONE_OVERLAP = os.environ.get("CORNER_ZONE_OVERLAP", "1") == "1"


def _zone_by_overlap(mask, ctr, h, w, fallback):
    """Quadrant holding the most of this component, not the one its
    centroid happens to fall in."""
    best, bz = 0, None
    for z in ("UR", "UL", "LL", "BR"):
        x0, y0, x1, y1 = _zone_box(z, ctr, h, w)
        v = int(mask[y0:y1, x0:x1].sum())
        if v > best:
            best, bz = v, z
    return bz if bz is not None else fallback
RECLAIM_MAX = 1200  # reclaimed-from-trace components above this are trace
CENTER_R = 28     # centroid closer than this to cell center = "center"
SURGERY_ENDS = int(os.environ.get("CORNER_SURGERY_ENDS", "30"))  # leftover gate: skeleton endpoints...
SURGERY_SKLEN = int(os.environ.get("CORNER_SURGERY_SKLEN", "90"))  # ...and skeleton length (see the surgery pass below). Both were tuned on three DIGIT targets (ends 40-70); a cursive S has fewer endpoints, so the endpoint half is the one that can be too strict for token marks.

ZONES = ["UR", "UL", "LL", "BR", "center"]


def zone_of(cxy, ctr):
    dx, dy = cxy[0] - ctr[0], cxy[1] - ctr[1]
    if dx * dx + dy * dy < CENTER_R * CENTER_R:
        return "center"
    if dy < 0:
        return "UR" if dx > 0 else "UL"
    return "BR" if dx > 0 else "LL"


def _zone_box(zone, ctr, h, w):
    x0 = ctr[0] if zone == "UR" else 0
    x1 = w if zone == "UR" else ctr[0]
    y0 = 0 if zone in ("UR", "UL") else ctr[1]
    y1 = ctr[1] if zone in ("UR", "UL") else h
    return x0, y0, x1, y1


def _through_path(sk):
    """BFS shortest path between the two farthest border skeleton
    pixels -- the trace's transit through the zone crop."""
    import collections
    h, w = sk.shape
    border = [(y, x) for y, x in zip(*np.nonzero(sk))
              if y < 2 or x < 2 or y >= h - 2 or x >= w - 2]
    if not WELD2:
        if len(border) < 2:
            return None
        best = None
        for a in border:
            for b in border:
                d = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                if best is None or d > best[0]:
                    best = (d, a, b)
        _, src, dst = best
    else:
        _, lab = cv2.connectedComponents(sk.astype(np.uint8), connectivity=8)
        cand = border if len(border) >= 2 else list(zip(*np.nonzero(sk)))
        best = None
        for a in cand:
            for b in cand:
                if lab[a] != lab[b]:
                    continue
                d = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                if best is None or d > best[0]:
                    best = (d, a, b)
        if best is None:
            return None
        _, src, dst = best
    prev = {src: None}
    q = collections.deque([src])
    while q:
        cur = q.popleft()
        if cur == dst:
            break
        y, x = cur
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if (0 <= ny < h and 0 <= nx < w and sk[ny, nx]
                        and (ny, nx) not in prev):
                    prev[(ny, nx)] = cur
                    q.append((ny, nx))
    if dst not in prev:
        return None
    path = np.zeros_like(sk)
    cur = dst
    while cur is not None:
        path[cur] = True
        cur = prev[cur]
    return path


# Width of the cut that erases the transit path. Fixed at 9 since the
# surgery was written, but the ink it cuts through is 3.5-4.6 px wide
# in every welded zone measured 2026-07-27, so a 9 px cut takes the
# glyph with the trace: two zones whose leftover fails the gate at 9
# pass it comfortably at 3.
SURGERY_K = int(os.environ.get("CORNER_SURGERY_K", "9"))
# Narrowing the cut globally was measured 2026-07-27 and is WORSE: at 3
# it recovers two more zones but leaves trace fragments welded to every
# other glyph, and the absolute correct count falls 324 -> 321 while
# GT-silent firings go 9 -> 27. The wide cut is load-bearing. What the
# same measurement shows, though, is that the two zones it loses clear
# the gate at 3 with enormous margin (sklen 114 and 251 against 90,
# ends 58 and 74 against 30). So the narrow cut is worth having only as
# a FALLBACK, entered when the wide cut leaves nothing, and only when
# the leftover is unambiguously a glyph.
SURGERY_RETRY_K = int(os.environ.get("CORNER_SURGERY_RETRY_K", "0"))
RETRY_SKLEN = int(os.environ.get("CORNER_RETRY_SKLEN", "100"))
RETRY_ENDS = int(os.environ.get("CORNER_RETRY_ENDS", "50"))
_K9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                (SURGERY_K, SURGERY_K))
_N8 = np.ones((3, 3), np.uint8)


def cell_clusters(resid, long_mask, x_play):
    h, w = resid.shape
    binr = ((resid >= edge_score.INK_ON) * 255).astype(np.uint8)
    binr[:, :x_play] = 0
    binr[long_mask > 0] = 0
    pad = cell_diff.PAD
    band = np.zeros_like(binr, bool)
    for yb in (pad, h - pad):
        band[max(0, yb - edge_score.BAND_HW):yb + edge_score.BAND_HW, :] = True
    for xb in (pad, w - pad, x_play - 3):
        band[:, max(0, xb - edge_score.BAND_HW):xb + edge_score.BAND_HW] = True
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * CLUSTER_GAP + 1, 2 * CLUSTER_GAP + 1))
    big = cv2.dilate(binr, k)
    n, labels, stats, cent = cv2.connectedComponentsWithStats(big)
    ctr = ((x_play + w) // 2, h // 2)
    out = []
    for i in range(1, n):
        comp = labels == i
        ink_px = np.count_nonzero(comp & (binr > 0))
        if ink_px < MIN_INK:
            continue
        if np.count_nonzero(comp & (binr > 0) & band) > 0.6 * ink_px:
            continue  # neighbor-cell bleed hugging a border
        x, y, ww, hh, _ = (int(v) for v in stats[i])
        z = zone_of(cent[i], ctr)
        if ZONE_OVERLAP and z != "center":
            z = _zone_by_overlap(comp & (binr > 0), ctr, h, w, z)
        out.append({
            "zone": z,
            "bbox": [x, y, ww, hh],
            "ink": int(ink_px),
        })

    # Reclaim pass (2026-07-12): a corner glyph written against the
    # base trace gets welded into the LONG component by score_cell's
    # 3x3 bridging dilation and wiped above -- 13 GT-labeled corners
    # had no crop at all for this reason. On the RAW undilated ink the
    # weld often disappears: re-segment the ink under long_mask and
    # reclaim compact, non-long components -- but only into corner
    # zones that have NO cluster yet, so all 279 previously measured
    # CORNER crops stay bit-identical (machine-verified) and the only
    # corner additions are zones that previously yielded nothing.
    # One caveat found in adversarial review: crop RECTANGLES can
    # straddle zones, so restoring reclaimed pixels into the shared
    # mask let exactly one center crop (g18:r6c2) see the restored --
    # real -- ink inside its rectangle; its label and the center LOO
    # (335/391 before and after) are unchanged, and hiding real ink
    # from a crop to preserve a hash would be the wrong trade. Probed on
    # the 13 absorbed corners: 5 separate this way ((5) and (3) fully
    # clean, a paren fragment, a partial 5, an S with trace residue);
    # the other 8 are true welds or genuinely glyph-free zones and
    # need path surgery, not re-segmentation. Measured end to end
    # (2026-07-12): corner samples 279 -> 284, occupancy 231/279 ->
    # 238/284 -- the old subset gains 4 with ZERO breaks (235/279,
    # template spillover plus richer per-cell zone info), 3 of the 5
    # new corners read correctly, and -- ironically -- the two fully
    # clean reclaimed glyphs are the two new misses. Center samples
    # unchanged (391 before and after).
    have = {c["zone"] for c in out}
    raw = ((resid >= edge_score.INK_ON) * 255).astype(np.uint8)
    raw[:, :x_play] = 0
    trace_ink = ((raw > 0) & (long_mask > 0)).astype(np.uint8)
    n2, lab2, st2, cent2 = cv2.connectedComponentsWithStats(trace_ink)
    for i in range(1, n2):
        x, y, ww, hh, area = (int(v) for v in st2[i])
        if max(ww, hh) >= edge_score.LONG_EXTENT:
            continue                      # still trace-sized
        if area < MIN_INK or area > RECLAIM_MAX:
            continue
        z = zone_of(cent2[i], ctr)
        if ZONE_OVERLAP and z != "center":
            z = _zone_by_overlap(lab2 == i, ctr, h, w, z)
        if z not in ("UR", "UL", "LL") or z in have:
            continue
        comp = lab2 == i
        if np.count_nonzero(comp & band) > 0.6 * area:
            continue                      # border bleed
        out.append({"zone": z, "bbox": [x, y, ww, hh], "ink": int(area),
                    "reclaimed": True})
        binr[comp] = 255                  # restore pixels for cropping

    # Weld-surgery pass (2026-07-12): the reclaim above only separates
    # glyphs the raw ink already disconnects from the trace; a glyph
    # drawn ON the trace stays one component and needs the trace cut
    # out. For corner zones STILL empty: skeletonize the zone's
    # long-mask ink, erase the through-path (BFS between the two
    # farthest border skeleton points, dilated to stroke width), and
    # keep what's left -- IF it passes the leftover-vs-arc gate. Fired
    # generically this surgery triggers 136 times for 3 real welded
    # glyphs (44:1 -- a rounded trace corner leaves compact leftovers),
    # which shelved it on 2026-07-12; the unlock is the shape gate
    # measured the same day on all 136 firings: glyph leftovers are
    # long ragged squiggles (skeleton length >= 90 with >= 30 endpoint
    # pixels; the three real targets sit at sklen 98-151 / ends 40-70)
    # while trace-arc leftovers are short smooth segments (median sklen
    # 31 / ends 4). The gate keeps 3/3 targets and 1/133 false firings
    # (g14:r7c1:UR, a near-straight trace fragment in a GT-silent
    # zone). Thresholds are corpus-tuned like MIN_INK -- revisit as GT
    # grows.
    have2 = {c["zone"] for c in out}
    todo = [z for z in ("UR", "UL", "LL") if z not in have2]
    if todo:
        import corner_knn as _ck   # lazy: corner_knn imports this module
    for z in todo:
        x0, y0, x1, y1 = _zone_box(z, ctr, h, w)
        ink = trace_ink[y0:y1, x0:x1] > 0
        if ink.sum() < MIN_INK:
            continue
        sk = _ck._zs_thin(ink)
        path = _through_path(sk)
        if path is None:
            continue
        def _cut(kw):
            K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kw))
            lf = ink & ~(cv2.dilate(path.astype(np.uint8), K) > 0)
            nl, labl, stl, _ = cv2.connectedComponentsWithStats(
                lf.astype(np.uint8))
            kp = np.zeros_like(lf)
            t = 0
            for i in range(1, nl):
                if stl[i][4] >= 25:
                    kp |= labl == i
                    t += int(stl[i][4])
            if t < MIN_INK:
                return kp, t, 0, 0
            ks = _ck._zs_thin(kp).astype(np.uint8)
            n8 = cv2.filter2D(ks, -1, _N8,
                              borderType=cv2.BORDER_CONSTANT) - ks
            return kp, t, int(ks.sum()), int(np.count_nonzero((ks > 0) & (n8 == 1)))

        keep, tot, sklen, ends = _cut(SURGERY_K)
        if (SURGERY_RETRY_K and
                (tot < MIN_INK or sklen < SURGERY_SKLEN or ends < SURGERY_ENDS)):
            k2, t2, s2, e2 = _cut(SURGERY_RETRY_K)
            if t2 >= MIN_INK and s2 >= RETRY_SKLEN and e2 >= RETRY_ENDS:
                keep, tot, sklen, ends = k2, t2, s2, e2
        if tot < MIN_INK:
            continue
        if sklen < SURGERY_SKLEN or ends < SURGERY_ENDS:
            continue                      # trace-arc leftover, not a glyph
        kys, kxs = np.nonzero(keep)
        out.append({"zone": z,
                    "bbox": [int(kxs.min()) + x0, int(kys.min()) + y0,
                             int(kxs.max() - kxs.min() + 1),
                             int(kys.max() - kys.min() + 1)],
                    "ink": tot, "surgery": True})
        binr[y0:y1, x0:x1][keep] = 255    # restore pixels for cropping
    return out, binr


def main():
    src, gridpath, ref, outpath = sys.argv[1:5]
    gtpath = sys.argv[5] if len(sys.argv) > 5 else None
    ref_rc = tuple(int(x) for x in ref.split(","))
    with open(gridpath) as f:
        grid = json.load(f)
    cols, rows = cell_diff.lattice(grid)
    gray = cv2.cvtColor(grid_detect.load_rotated(src), cv2.COLOR_BGR2GRAY)
    aligned, template, _ = cell_diff.build_aligned(gray, cols, rows, ref_rc)
    x_play = cell_diff.PAD + cell_diff.strip_width(grid["v_clusters"]) + 3

    report = {}
    for (ri, ci) in sorted(aligned.keys()):
        resid = cell_diff.residual(template, aligned[(ri, ci)])
        s, long_mask = edge_score.score_cell(resid, x_play)
        cl, _ = cell_clusters(resid, long_mask, x_play)
        if cl or s["ink"] >= 80:
            report[f"r{ri}c{ci}"] = cl
    with open(outpath, "w") as f:
        json.dump(report, f, indent=1)
    print(f"{len(report)} cells with clusters -> {outpath}")

    if not gtpath:
        return
    with open(gtpath) as f:
        gt = json.load(f)
    corner_gt = {"UR": "2B", "UL": "3B", "LL": "home"}
    hit = miss = extra = 0
    for tag, truth in sorted(gt["cells"].items()):
        if not truth.get("pa"):
            continue
        zones = {c["zone"] for c in report.get(tag, [])}
        adv = truth.get("advances", {})
        for z, base in corner_gt.items():
            want = base in adv
            got = z in zones
            if want and got:
                hit += 1
            elif want and not got:
                miss += 1
                print(f"  {tag}: {base} advance written but no {z} cluster")
            elif got and not want:
                extra += 1
                print(f"  {tag}: {z} cluster but no {base} advance in GT")
    print(f"corner clusters vs GT advances: hit={hit} miss={miss} "
          f"extra={extra}")


if __name__ == "__main__":
    main()
