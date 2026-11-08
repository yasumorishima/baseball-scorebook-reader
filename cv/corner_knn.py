#!/usr/bin/env python3
"""Corner-number and center-mark reader, evaluated leave-one-out.

The ground truth labels every zone implicitly: advances{"2B":4} means
the upper-right cluster of that cell reads "(4)", an out cell's center
reads its out numeral, a scored cell's center reads the run circle. So
the transcribed game doubles as a labeled training set for a template
matcher -- same scorer, same hand, so nearest-neighbor over normalized
zone crops should carry most of the recognition weight.

This tool harvests (crop, label) pairs from a game + its ground truth
and reports leave-one-out 1-NN accuracy (dice distance on 40x40
binarized zone crops). Singleton classes cannot be right under LOO and
are reported separately, not hidden in the average.

Usage: corner_knn.py <image> <grid.json> <gt.json> <ref_r,ref_c> [dumpdir]
       corner_knn.py <games-manifest.json> [dumpdir]
       (manifest = list of {id, image, grid, gt, ref} -- pools every
        game's samples into one template set before the LOO)
"""
import cv2
import numpy as np
import json
import os
import re
import sys

import annot_zones
import cell_diff
import edge_score
import grid_detect

NORM = 40  # normalized crop side
WIDTH_NORM = os.environ.get("CORNER_WIDTH_NORM", "0") == "1"
_W3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def _zs_thin(img):
    """Zhang-Suen thinning to a 1px skeleton (vectorized numpy)."""
    img = img.astype(np.uint8)
    while True:
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            n2, n3 = p[:-2, 1:-1], p[:-2, 2:]
            n4, n5 = p[1:-1, 2:], p[2:, 2:]
            n6, n7 = p[2:, 1:-1], p[2:, :-2]
            n8, n9 = p[1:-1, :-2], p[:-2, :-2]
            ring = [n2, n3, n4, n5, n6, n7, n8, n9, n2]
            b = n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9
            a = sum(((ring[k] == 0) & (ring[k + 1] == 1)).astype(np.uint8)
                    for k in range(8))
            if step == 0:
                cond = ((img == 1) & (b >= 2) & (b <= 6) & (a == 1) &
                        (n2 * n4 * n6 == 0) & (n4 * n6 * n8 == 0))
            else:
                cond = ((img == 1) & (b >= 2) & (b <= 6) & (a == 1) &
                        (n2 * n4 * n8 == 0) & (n2 * n6 * n8 == 0))
            if cond.any():
                img[cond] = 0
                changed = True
        if not changed:
            return img > 0


def normalize(mask):
    """Tight-crop, pad square, resize to NORM x NORM binary.

    CORNER_WIDTH_NORM=1 additionally reduces the resized crop to its
    skeleton and re-inks it at a fixed 3px stroke. MEASURED AND
    REJECTED as a default (2026-07-04, three games): corner LOO 33->35%
    but end-to-end occupancy 82->78% -- stroke width is a real cue
    (cursive S/PB squiggles are drawn heavier and faster than careful
    paren digits), and flattening it makes the S/PB-vs-faint-digit
    confusion WORSE where it matters. Per-sheet threshold calibration
    was probed the same day and rejected too: the aligned-median
    residuals already put all three sheets on near-identical intensity
    distributions (medians 45-49, Otsu 51-62), so there is no per-sheet
    faintness left to correct. The flag stays for cheap re-measurement
    when more GT arrives; leave it off."""
    ys, xs = np.nonzero(mask)
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = m.shape
    side = max(h, w)
    sq = np.zeros((side, side), np.uint8)
    sq[(side - h) // 2:(side - h) // 2 + h,
       (side - w) // 2:(side - w) // 2 + w] = m
    out = cv2.resize(sq, (NORM, NORM), interpolation=cv2.INTER_AREA) > 0
    if WIDTH_NORM:
        out = cv2.dilate(_zs_thin(out).astype(np.uint8), _W3) > 0
    return out


SOFT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def soften(crop):
    """Dilate the normalized crop: raw dice on 3-5px strokes drops to
    near-zero overlap under a few px of misalignment (measured 5/27 LOO
    on the reference game); the softened version tolerates it."""
    return cv2.dilate(crop.astype(np.uint8), SOFT) > 0


def dice_dist(a, b):
    inter = np.count_nonzero(a & b)
    return 1.0 - 2.0 * inter / (np.count_nonzero(a) + np.count_nonzero(b))


def holes(mask):
    """Enclosed background regions >= 4px (6 has one, 8 two, 2/3/5/7 none)."""
    inv = (~mask).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(inv)
    h, w = mask.shape
    cnt = 0
    for j in range(1, n):
        x, y, ww, hh, area = (int(v) for v in st[j])
        if area >= 4 and x > 0 and y > 0 and x + ww < w and y + hh < h:
            cnt += 1
    return cnt


def digit_view(crop):
    """Middle of the corner glyph: (n)'s parens hug the left/right edges
    of the normalized crop, so blanking the outer ~22% isolates the
    digit -- whole-glyph matching wastes most of its overlap on parens
    shared by every class."""
    m = crop.copy()
    m[:, :9] = False
    m[:, 31:] = False
    if not m.any():
        return crop
    return normalize(m)


HOLE_W = 0.35  # per-hole-difference penalty added to the dice distance
FULL_W = float(os.environ.get("CORNER_FULL_W", "0.3"))  # whole-crop dice term weight (corner distance only)


def _corner_feats_one(crops):
    """Distance features for corner crops.

    Three terms: the paren-stripped digit view (shape of the numeral),
    the hole count, and the FULL crop. The full-crop term restores the
    notation cue that digit_view deliberately throws away: numbered
    advances are always written as (n) WITH parens while S/PB are bare,
    so whole-glyph overlap separates the classes the digit view cannot.
    Measured 2026-07-11 on the complete 25-game corpus (279 corners),
    FULL_W grid {0.3, 0.5, 1.0}: raw top-1 41% -> 44-45% at EVERY
    weight (the robust effect) and top-3 66% -> 67-69%; the solver's
    constrained/joint stages improve too (53->56-58%, 63->66-67%). The
    final occupancy stage however reshuffles whole-cell joint
    assignments, so its movement is churn-dominated: 223 baseline ->
    225 (0.3) / 220 (0.5) / 224 (1.0) out of 279, i.e. the end-to-end
    +2 at the 0.3 default sits within a ~5-cell churn amplitude and is
    provisional; the recognition-stage gain is the real result. Final
    S/PB-boundary misses 32 -> 28 at 0.3 (ledger: +12 repaired, -10
    broken). A connected-component-count penalty was probed the same
    day and REJECTED (S is 1 component in 18/20 but so are 58/243 faint
    digits; every weight tried lost top-1 accuracy), as was a bare
    side-band paren-ink fraction (multi-cluster crops center the glyph
    unpredictably; no class separation)."""
    views = [digit_view(c) for c in crops]
    return ([soften(v) for v in views], [holes(v) for v in views],
            [soften(c) for c in crops])


def _corner_dist_one(feats, i, j):
    softd, nh, softf = feats
    return (dice_dist(softd[i], softd[j]) + HOLE_W * abs(nh[i] - nh[j])
            + FULL_W * dice_dist(softf[i], softf[j]))


CHAMFER = os.environ.get("CORNER_CHAMFER", "1") == "1"
CHAM_HOLE_W = float(os.environ.get("CORNER_CHAM_HOLE_W", "0.10"))
CHAM_FULL_W = float(os.environ.get("CORNER_CHAM_FULL_W", "0.45"))
CHAM_ORIENT_W = float(os.environ.get("CORNER_CHAM_ORIENT_W", "0.80"))
CHAM_BINS = int(os.environ.get("CORNER_CHAM_BINS", "10"))
CHAM_CLIP = float(os.environ.get("CORNER_CHAM_CLIP", "20"))
CHAM_SIGMA = float(os.environ.get("CORNER_CHAM_SIGMA", "0.6"))
CHAM_FULL_ORIENT = os.environ.get("CORNER_CHAM_FULL_ORIENT", "1") == "1"
# Elastic block matching. The oriented term still lets every ink pixel
# find its own nearest same-tangent template pixel independently, so a
# glyph can pay its way across a template whose strokes sit in quite
# different places. Splitting the crop into a GRID x GRID mesh and
# allowing each block its own small translation makes the displacement
# locally COHERENT: a block moves as a unit or not at all. 0 disables.
CHAM_GRID = int(os.environ.get("CORNER_CHAM_GRID", "0"))
CHAM_SHIFT = int(os.environ.get("CORNER_CHAM_SHIFT", "1"))
# Whether the elastic mesh also applies to the whole-glyph notation
# term. That term exists to separate parenthesized digits from bare
# S/PB, and deformation freedom is exactly what could let a bare mark
# stretch into a parenthesized one, so it gets its own switch.
CHAM_GRID_FULL = os.environ.get("CORNER_CHAM_GRID_FULL", "1") == "1"
# Chamfer core instead of dice-after-dilate. soften() exists because raw
# dice collapses under a few px of stroke misalignment; charging the
# ACTUAL displacement via a distance transform is the continuous form of
# that patch, so the term that was already carrying the pipeline gets
# made honest rather than replaced. Set CORNER_CHAMFER=0 to reproduce
# every measurement taken before 2026-07-26 bit-identically (verified
# twice: the 295/346 miss list does not move by a single line).
#
# Measured 2026-07-26, 29 sheets / 346 corners. Representation alone,
# everything else held fixed (same views, same min-over-views, same
# LOO m-NN=2, no hole term, no FULL term): dice 198 vs chamfer 209 raw,
# digits 177 vs 188, S/PB 21/38 unchanged -- chamfer core by itself
# already beats the whole dice pipeline (204).
#
# End to end at CHAM_HOLE_W=0.10, sweeping CHAM_FULL_W:
#   0.15 raw 226 con 259 joint 278 occ 296
#   0.30 raw 225 con 261 joint 280 occ 297
#   0.50 raw 223 con 260 joint 280 occ 298
#   0.70 raw 218 con 257 joint 278 occ 303
#   1.00 raw 215 con 257 joint 274 occ 305  <- default
#   1.20 raw 211 con 253 joint 271 occ 303
#   1.50 raw 207 con 244 joint 261 occ 297
# vs the dice baseline raw 204 con 243 joint 265 occ 295. Every stage
# improves at the default, occupancy 85% -> 88%, ledger 13 repaired /
# 3 broken, selfcheck PASS on all 29 games. The occupancy optimum sits
# on a 303-305 plateau over FULL_W 0.70-1.20, not on a spike, so the
# default is the middle of a flat region rather than a fitted peak.
#
# Note the shape of that sweep: raw FALLS from 226 to 215 while
# occupancy RISES from 296 to 305. The FULL term is the one that
# separates bare S/PB from paren digits, and the solver treats those two
# as structurally different (a digit is a batter slot and takes an
# agenda position, S/PB is play-unknown and does not). So paying
# recognition accuracy for a cleaner S/PB boundary is a net win at the
# stage that matters -- which digit is a weaker question than whether it
# is a digit at all. Untested corollary worth probing next: the
# remaining block may be the S/PB boundary rather than digit shape.
# HOLE_W was swept too (0.00/0.10/0.20/0.35); 0.10 wins at every FULL_W.


def _orient_bins(mask):
    """Per-pixel stroke-tangent bin for an ink mask.

    Chamfer charges displacement but not direction, so any ink can pay
    its way to any other ink -- exactly the freedom behind the residual
    digit-vs-digit confusions. The tangent is the gradient of the
    blurred mask turned ninety degrees, taken modulo pi because strokes
    are undirected.
    """
    f = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), CHAM_SIGMA)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    ang = (np.arctan2(gy, gx) + np.pi / 2) % np.pi
    b = (ang / (np.pi / CHAM_BINS)).astype(np.int32)
    return np.minimum(b, CHAM_BINS - 1)


def _chamfer_feats_one(crops, view=None):
    """Ink masks, their distance transforms, hole counts, and -- when
    the oriented term is on -- per-tangent-bin ink indices and clipped
    per-bin distance transforms. `view` selects the mask: the default
    paren-stripped digit view for the shape core, or identity for the
    whole-glyph notation term."""
    digits = [(view or digit_view)(c) for c in crops]
    dt = [cv2.distanceTransform((~d).astype(np.uint8), cv2.DIST_L2, 3)
          for d in digits]
    cnt = [max(1, int(np.count_nonzero(d))) for d in digits]
    oidx, odt, mesh = [], [], []
    if CHAM_ORIENT_W > 0:
        S, G = CHAM_SHIFT, CHAM_GRID
        for d in digits:
            h, w = d.shape
            b = np.where(d, _orient_bins(d), -1).ravel()
            oidx.append([np.flatnonzero(b == t) for t in range(CHAM_BINS)])
            per, padded = [], []
            for t in range(CHAM_BINS):
                m = (b == t).reshape(d.shape)
                if not m.any():
                    dtb = np.full(d.shape, CHAM_CLIP, np.float32)
                else:
                    dtb = np.minimum(cv2.distanceTransform(
                        (~m).astype(np.uint8), cv2.DIST_L2, 3), CHAM_CLIP)
                per.append(dtb.ravel())
                if G > 0:
                    padded.append(np.pad(dtb, S, constant_values=CHAM_CLIP))
            odt.append(per)
            if G <= 0:
                continue
            # one stacked lookup table so a pair costs two numpy calls per
            # shift instead of two per (shift, tangent bin)
            pw = w + 2 * S
            stack = np.stack(padded).ravel()
            ys, xs = np.nonzero(d)
            tt = b.reshape(d.shape)[ys, xs]
            flat = (tt.astype(np.int64) * (h + 2 * S) * pw
                    + (ys + S) * pw + (xs + S))
            blk = (ys * G // h) * G + (xs * G // w)
            mesh.append((stack, flat, blk.astype(np.int64), pw))
    return digits, dt, cnt, [holes(d) for d in digits], oidx, odt, mesh


def _cham_pair(cham, k, i, j, grid=True):
    ink, dt, cnt = cham[k][0], cham[k][1], cham[k][2]
    plain = (float(dt[j][ink[i]].sum()) / cnt[i]
             + float(dt[i][ink[j]].sum()) / cnt[j])
    if CHAM_ORIENT_W <= 0:
        return plain
    oidx, odt, mesh = cham[k][4], cham[k][5], cham[k][6]

    if CHAM_GRID > 0 and grid:
        S, nb = CHAM_SHIFT, CHAM_GRID * CHAM_GRID

        def one(a, b):
            stack = mesh[b][0]
            flat, blk, pw = mesh[a][1], mesh[a][2], mesh[b][3]
            if flat.size == 0:
                return 0.0
            best = None
            for dy in range(-S, S + 1):
                for dx in range(-S, S + 1):
                    v = stack[flat + dy * pw + dx]
                    sums = np.bincount(blk, weights=v, minlength=nb)
                    best = sums if best is None else np.minimum(best, sums)
            return float(best.sum()) / cnt[a]
    else:
        def one(a, b):
            tot = 0.0
            for t in range(CHAM_BINS):
                ix = oidx[a][t]
                if ix.size:
                    tot += float(odt[b][t][ix].sum())
            return tot / cnt[a]

    return ((1.0 - CHAM_ORIENT_W) * plain
            + CHAM_ORIENT_W * (one(i, j) + one(j, i)))


def corner_feats(view_lists, scale_mask=None):
    """Per-view feature sets over multi-view corner samples.

    scale_mask: optional per-sample bool. When given, ONLY the samples
    it marks contribute to the chamfer scale factors. Those factors are
    a pool-wide average of pair distances, so an unlabelled sample that
    can never be a template still moves the metric for every labelled
    one -- measured 2026-08-30, the mark detector's 178 phantom crops
    cost raw recognition 5 corners that way. Passing an all-True mask is
    identical to passing none.

    Each sample is a list of NVIEWS candidate crops of the same zone
    (see zone_views); features are built per view slot, and the pair
    distance is the MIN across view slots (both sides using the same
    slot). Crop hygiene measured 2026-07-11 (279 corners, zone-wise
    occupancy misses had concentrated on the home corner at 65%): the
    legacy 12px-merge union crop drags arrows / w.p squiggles /
    underlines into the glyph box, and single-cluster views fix exactly
    that -- LOO top-1 union 45% / seed-only 53% / compact-only 54% /
    MIN over the three 56% (top-3 74%, home corner 40%->57%). The union
    view stays in the set because paren+digit sometimes split into
    separate clusters that only the merge reunites."""
    base = [_corner_feats_one([v[k] for v in view_lists])
            for k in range(NVIEWS)]
    if not CHAMFER:
        return base
    cham = [_chamfer_feats_one([v[k] for v in view_lists])
            for k in range(NVIEWS)]
    n = len(view_lists)
    idx = ([i for i in range(n) if scale_mask[i]]
           if scale_mask is not None else list(range(n)))
    if not idx:                       # nothing labelled: keep the pool
        idx = list(range(n))
    sub = min(len(idx), 60)

    def _scale(fs, k, grid=True):
        tot = 0.0
        for i in idx[:sub]:
            tot += (sum(_cham_pair(fs, k, i, j, grid) for j in idx)
                    / len(idx))
        return tot / sub if sub else 1.0

    scales = [_scale(cham, k) for k in range(NVIEWS)]
    if not CHAM_FULL_ORIENT:
        return (base, cham, scales, None, 1.0)
    # The notation term compares whole glyphs -- parens included -- and
    # is the last part of the distance still measured by binary overlap.
    full = [_chamfer_feats_one([v[0] for v in view_lists], view=lambda c: c)]
    return (base, cham, scales, full, _scale(full, 0, CHAM_GRID_FULL))


def corner_dist(feats, i, j):
    if CHAMFER:
        base, cham, scales, full, fscale = feats
        core = None
        for k in range(NVIEWS):
            nh = cham[k][3]
            d = (_cham_pair(cham, k, i, j) / scales[k]
                 + CHAM_HOLE_W * abs(nh[i] - nh[j]))
            core = d if core is None else min(core, d)
        if full is not None:
            return core + CHAM_FULL_W * (
                _cham_pair(full, 0, i, j, CHAM_GRID_FULL) / fscale)
        softf = base[0][2]
        return core + CHAM_FULL_W * dice_dist(softf[i], softf[j])
    """Pairwise corner distance: the digit-shape core (digit view +
    holes) takes the MIN over the parallel view slots, but the FULL-crop
    term is always computed on the union view (slot 0). The full term is
    the whole-glyph notation cue that separates bare S/PB from paren
    digits -- letting it float to fragment views destroys exactly that
    identity: measured 2026-07-11, plain min-over-views lifted digits
    105->143/252 but collapsed S recall 14->10/20 and PB 6->3/7 (and the
    S/PB gate needs top-1), while this hybrid keeps S at 14/20, digits
    at 131/252, and the home corner at its best 49/81."""
    core = min(dice_dist(f[0][i], f[0][j]) + HOLE_W * abs(f[1][i] - f[1][j])
               for f in feats)
    softf = feats[0][2]
    return core + FULL_W * dice_dist(softf[i], softf[j])


def _crop_boxes(annot_mask, boxes):
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    crop = annot_mask[y0:y1, x0:x1] > 0
    if np.count_nonzero(crop) < annot_zones.MIN_INK:
        return None
    return normalize(crop)


def zone_crop(clusters, annot_mask, zone):
    """Union of the zone's clusters near its biggest one -- stray specks
    elsewhere in the quadrant would stretch the bbox and break the
    digit-in-the-middle assumption."""
    zc = [c for c in clusters if c["zone"] == zone]
    if not zc:
        return None
    seed = max(zc, key=lambda c: c["ink"])["bbox"]
    boxes = [c["bbox"] for c in zc if
             c["bbox"][0] < seed[0] + seed[2] + 12 and
             seed[0] < c["bbox"][0] + c["bbox"][2] + 12 and
             c["bbox"][1] < seed[1] + seed[3] + 12 and
             seed[1] < c["bbox"][1] + c["bbox"][3] + 12]
    return _crop_boxes(annot_mask, boxes)


NVIEWS = 4  # union / biggest-cluster / most-compact-cluster / de-ringed
RING_COV = float(os.environ.get("CORNER_RING_COV", "0.7"))


def _de_ring(mask):
    """Erase a dominant circular arc from a normalized crop, or None
    when no ring is found.

    The home corner's run circle (and the circled-digit reach notation
    of some scorers) fuses with the digit into a single component, so
    the component-level ring peel probed 2026-07-11 never fired (3-5 of
    279). This works at stroke level instead: fit the min enclosing
    circle of the biggest outer contour, measure how much of that
    circle's circumference carries ink (15-degree bins), and when the
    arc coverage clears RING_COV erase the annulus band and re-normalize
    what remains -- the interior digit, zoomed back to full frame.
    Coverage distribution measured on the 25-game corpus (279 corners):
    fused rings sit at 0.75-1.00, plain paren digits p75=0.62, S
    squiggles mean 0.28, so 0.7 fires on 47/279 (S recall untouched).

    Measured 2026-07-11: corner LOO top-1 149 -> 151/279, false S/PB
    15 -> 14, S 14/20 and PB 4/7 unchanged; end-to-end occupancy
    230 -> 231/279 with the miss list otherwise bit-identical (repaired
    007:r1c0:UR, an S-siphon whose digit neighbor got closer through
    the neighbor's de-ringed slot; zero churn). Probed the same day and
    REJECTED: replacing (not min-ing) the core with the de-ringed slot
    for ring-pairs (147 -- fragment-vs-fragment dice is noisier than
    the shared-ring overlap it was meant to fix), a wider 0.20*rad band
    (148), dropping ink outside the circle too (146), and RING_COV
    0.65/0.75/0.8 (equal or -1 vs 0.7)."""
    m8 = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    big = max(cnts, key=cv2.contourArea)
    (cx, cy), rad = cv2.minEnclosingCircle(big)
    if rad < 6:
        return None
    ys, xs = np.nonzero(m8)
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    near = np.abs(d - rad) < max(3.0, 0.15 * rad)
    if not near.any():
        return None
    ang = (np.degrees(np.arctan2(ys[near] - cy, xs[near] - cx))
           // 15).astype(int)
    if len(set(ang.tolist())) / 24.0 < RING_COV:
        return None
    keep = m8.astype(bool).copy()
    keep[ys[near], xs[near]] = False
    if np.count_nonzero(keep) < annot_zones.MIN_INK:
        return None
    return normalize(keep)


def _compactness(c):
    """Dense, square-ish clusters look like paren digits; arrows,
    underlines and w.p squiggles are elongated or sparse."""
    x, y, w, h = c["bbox"]
    if w == 0 or h == 0:
        return -1.0
    fill = c["ink"] / float(w * h)
    elong = max(w, h) / max(1.0, min(w, h))
    return fill / elong


def zone_views(clusters, annot_mask, zone):
    """NVIEWS candidate crops of the zone glyph, or None when the zone
    is absent/absorbed (same gate as zone_crop, so the sample set is
    unchanged). Slots: 0 = legacy 12px-merge union, 1 = biggest cluster
    alone, 2 = most compact cluster alone, 3 = de-ringed union (fused
    score-circle surgery); a slot that falls under MIN_INK (or finds no
    ring) falls back to the union so every sample carries NVIEWS."""
    union = zone_crop(clusters, annot_mask, zone)
    if union is None:
        return None
    zc = [c for c in clusters if c["zone"] == zone]
    seed = _crop_boxes(annot_mask, [max(zc, key=lambda c: c["ink"])["bbox"]])
    comp = _crop_boxes(annot_mask, [max(zc, key=_compactness)["bbox"]])
    dering = _de_ring(union)
    return [union, seed if seed is not None else union,
            comp if comp is not None else union,
            dering if dering is not None else union]


def load_game(src, gridpath, ref_rc):
    """Aligned cells + geometry for one sheet."""
    with open(gridpath) as f:
        grid = json.load(f)
    cols, rows = cell_diff.lattice(grid)
    gray = cv2.cvtColor(grid_detect.load_rotated(src), cv2.COLOR_BGR2GRAY)
    aligned, template, _ = cell_diff.build_aligned(gray, cols, rows, ref_rc)
    x_play = cell_diff.PAD + cell_diff.strip_width(grid["v_clusters"]) + 3
    return aligned, template, x_play


def harvest(gt, aligned, template, x_play):
    """(crop, label, tag) sample lists keyed by kind, labeled from GT."""
    corner_gt = {"UR": "2B", "UL": "3B", "LL": "home"}
    samples = {"corner": [], "center": []}
    for tag, truth in sorted(gt["cells"].items()):
        if not truth.get("pa"):
            continue
        ri, ci = (int(x) for x in
                  re.fullmatch(r"r(\d+)c(\d+)", tag).groups())
        resid = cell_diff.residual(template, aligned[(ri, ci)])
        _, long_mask = edge_score.score_cell(resid, x_play)
        clusters, annot_mask = annot_zones.cell_clusters(
            resid, long_mask, x_play)
        adv = truth.get("advances", {})
        for zone, base in corner_gt.items():
            if base not in adv:
                continue
            views = zone_views(clusters, annot_mask, zone)
            if views is None:
                print(f"  {tag} {zone}: labeled ({adv[base]}) but no crop "
                      f"(absorbed into trace)")
                continue
            samples["corner"].append(
                (views, str(adv[base]), f"{tag}:{zone}"))
        out = truth.get("out")
        label = None
        if out and out.get("seq"):
            label = "I" * out["seq"]           # I / II / III
        elif truth.get("scored"):
            label = "o"                        # run circle
        if label:
            crop = zone_crop(clusters, annot_mask, "center")
            if crop is not None:
                samples["center"].append((crop, label, f"{tag}:center"))
    return samples


def load_samples(games):
    """Pooled (crop, label, "gameid:tag:zone") samples over a manifest."""
    samples = {"corner": [], "center": []}
    for g in games:
        with open(g["gt"]) as f:
            gt = json.load(f)
        ref_rc = tuple(int(x) for x in g["ref"].split(","))
        aligned, template, x_play = load_game(g["image"], g["grid"], ref_rc)
        ss = harvest(gt, aligned, template, x_play)
        for kind, lst in ss.items():
            samples[kind].extend(
                (c, lab, f"{g['id']}:{t}") for c, lab, t in lst)
    return samples


def main():
    if len(sys.argv) >= 5:   # single game, legacy CLI
        src, gridpath, gtpath, ref = sys.argv[1:5]
        dumpdir = sys.argv[5] if len(sys.argv) > 5 else None
        games = [{"id": "g0", "image": src, "grid": gridpath,
                  "gt": gtpath, "ref": ref}]
    else:                    # games manifest
        with open(sys.argv[1]) as f:
            games = json.load(f)
        dumpdir = sys.argv[2] if len(sys.argv) > 2 else None
    samples = load_samples(games)

    for kind, ss in samples.items():
        counts = {}
        for _, lab, _ in ss:
            counts[lab] = counts.get(lab, 0) + 1
        if kind == "corner":
            feats = corner_feats([c for c, _, _ in ss])
        else:
            views = [c for c, _, _ in ss]
            nh = [0] * len(ss)  # hole counts are noise on center marks
                                # (thick pencil circles fill in; measured
                                # center LOO 78% -> 65% with them on)
            soft = [soften(v) for v in views]
        n_ok = n_bad = n_single = 0
        top3_ok = 0
        for i, (crop, lab, src_tag) in enumerate(ss):
            if counts[lab] < 2:
                n_single += 1
                continue
            if kind == "corner":
                dists = sorted(
                    (corner_dist(feats, i, j), lab2)
                    for j, (_, lab2, _) in enumerate(ss) if i != j)
            else:
                dists = sorted(
                    (dice_dist(soft[i], soft[j]) +
                     HOLE_W * abs(nh[i] - nh[j]), lab2)
                    for j, (_, lab2, _) in enumerate(ss) if i != j)
            # ranked label shortlist: nearest distance per distinct label
            shortlist = []
            for d, lab2 in dists:
                if lab2 not in shortlist:
                    shortlist.append(lab2)
            best = shortlist[0]
            if lab in shortlist[:3]:
                top3_ok += 1
            if best == lab:
                n_ok += 1
            else:
                n_bad += 1
                print(f"  {kind} {src_tag}: true={lab} pred={best} "
                      f"top3={shortlist[:3]} (d={dists[0][0]:.2f})")
            if dumpdir:
                os.makedirs(dumpdir, exist_ok=True)
                img = crop[0] if kind == "corner" else crop
                cv2.imwrite(
                    f"{dumpdir}/{kind}_{lab}_{src_tag.replace(':', '_')}.png",
                    img.astype(np.uint8) * 255)
        n = n_ok + n_bad
        acc = f"{n_ok}/{n} = {n_ok / n:.0%}" if n else "n/a"
        t3 = f" top3 {top3_ok}/{n} = {top3_ok / n:.0%}" if n else ""
        print(f"{kind}: LOO accuracy {acc}{t3} "
              f"(+{n_single} singleton labels excluded) classes={counts}")


if __name__ == "__main__":
    main()
