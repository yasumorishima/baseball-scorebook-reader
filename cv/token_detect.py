#!/usr/bin/env python3
"""Token detection: which accepted corners are marks WITHOUT a slot
number?

The mark detector is right about these corners -- the scorer did write
something there -- but the notation has two kinds of corner mark and
only one of them names a batter. `(4)` says "this runner reached the
base during slot 4's plate appearance"; PB, S, WP, 1E/2E/3E, DB, T.O
say the runner advanced with no batter to charge it to, and the ground
truth files them under `annotations` for exactly that reason. The combo
space can only read a corner as a numbered arrival, so an unnumbered
one pins the wrong batter and pushes the real corner in the same cell
onto the next one. Measured 2026-09-02: of the 17 corners that are
right with staged marks and wrong with the detector, 11 are a home mark
reading one slot too high with an unnumbered token sitting in UL.

Why this is a separate recogniser and not another chamfer label. Given
the tokens their own label class inside the existing metric, top-1
admits 108 of 178 tokens but siphons 70 of 351 real marks and occupancy
falls 311 -> 262: a 178-member class wins any 2-nearest race against
one-digit classes holding a dozen templates each, so the failure is
class size, not shape. Requiring UNC to win by a margin only walks that
back (0.05 -> 70 admitted / 38 siphoned / 289).

What the chamfer distance blurs is describable in a handful of numbers:
`(4)` is a digit flanked by two thin tall arcs, PB/WP/DB are letter
pairs, and the rest are single strokes and blobs. So: component count,
enclosed holes, column gaps, how tall the ink stands at the left and
right edges, aspect, density, absolute size, the zone, and the mark
detector's own margin. Leave-one-SHEET-out, the same fold pa_detect
uses and for the same reason -- one sheet is one hand, one pen, one
exposure.

Operating point. The two ends of the oracle measure the exchange rate:
admitting a token is worth about 14/79 = 0.18 corners, siphoning a real
mark costs about 49/70 = 0.7, so the ledger needs roughly four
admissions per siphon. The sheet-fold sweep gives 14:1 at 1.0 and
10.5:1 at 0.7, falling to 3.5:1 at 0.5 -- so the threshold sits high on
purpose and admits a minority of the tokens. Like every threshold in
this pipeline it was read off the same 31 sheets it is scored on.
"""
import os

import cv2
import numpy as np

ZONES = ("UR", "UL", "LL")
KEYS = ("comps", "holes", "gaps", "span_l", "span_r", "dens", "aspect",
        "top_frac", "colmax", "rowmax", "z_ink", "z_n", "box_w", "box_h",
        "box_ar", "margin", "is_UR", "is_UL", "is_LL")


def zone_stats(clusters, zone):
    """ink / cluster count / biggest box of one zone of one cell."""
    ink = n = mx = w = h = 0
    for c in clusters:
        if c["zone"] != zone:
            continue
        ink += int(c["ink"])
        n += 1
        if c["ink"] > mx:
            mx = int(c["ink"])
            w, h = int(c["bbox"][2]), int(c["bbox"][3])
    return {"ink": ink, "n": n, "w": w, "h": h}


def crop_feats(crop):
    a = np.asarray(crop)
    if a.dtype != bool:
        a = a > 0
    h, w = a.shape
    n, _lab, st, _c = cv2.connectedComponentsWithStats(
        a.astype(np.uint8), connectivity=8)
    comps = sorted((int(st[i][4]) for i in range(1, n)), reverse=True)
    big = [c for c in comps if c >= 8]
    inv = (~a).astype(np.uint8)
    nn, _l2, stv, _c2 = cv2.connectedComponentsWithStats(inv, connectivity=4)
    hole = 0
    for i in range(1, nn):
        x, y, ww, hh, area = (int(v) for v in stv[i])
        if x > 0 and y > 0 and x + ww < w and y + hh < h and area >= 4:
            hole += 1
    colink = a.sum(axis=0)
    rowink = a.sum(axis=1)
    gaps = 0
    prev = colink[0] > 0
    for c in colink[1:]:
        cur = c > 0
        if prev and not cur:
            gaps += 1
        prev = cur
    band = max(1, int(round(w * 0.2)))

    def span(sl):
        ys = np.nonzero(sl.any(axis=1))[0]
        return (ys[-1] - ys[0] + 1) / float(h) if len(ys) else 0.0

    return {
        "comps": len(big),
        "holes": hole,
        "gaps": gaps,
        "span_l": span(a[:, :band]),
        "span_r": span(a[:, -band:]),
        "dens": float(a.sum()) / (h * w),
        "aspect": w / float(h),
        "top_frac": float(comps[0]) / max(1, a.sum()) if comps else 0.0,
        "colmax": float(colink.max()) / h,
        "rowmax": float(rowink.max()) / w,
    }


def feats(view, zstat, zone, margin):
    f = crop_feats(view)
    f.update({
        "z_ink": zstat.get("ink", 0),
        "z_n": zstat.get("n", 0),
        "box_w": zstat.get("w", 0),
        "box_h": zstat.get("h", 0),
        "box_ar": zstat.get("w", 0) / float(max(1, zstat.get("h", 0))),
        "margin": float(margin),
    })
    for z in ZONES:
        f["is_" + z] = 1.0 if zone == z else 0.0
    return f


def _matrix(fs):
    return np.array([[float(f[k]) for k in KEYS] for f in fs], float)


def predict(samples, mnn=None, thr=None):
    """samples: [(gid, tag, zone, features, is_token)]
    -> {(gid, tag, zone): True if it should carry an unconstrained
    label}. Standardisation is refit inside every fold, so nothing
    about the held-out sheet reaches the rule that judges it."""
    mnn = int(os.environ.get("CORNER_TOKEN_MNN", "3")) if mnn is None else mnn
    thr = (float(os.environ.get("CORNER_TOKEN_THR", "0.7"))
           if thr is None else thr)
    X = _matrix([f for _, _, _, f, _ in samples])
    y = np.array([t for _, _, _, _, t in samples], bool)
    gid = np.array([g for g, _, _, _, _ in samples])
    out = {}
    if len(set(gid)) < 2 or not y.any() or y.all():
        # one sheet, or one class: the fold is empty and every distance
        # is nan. pa_detect hit the same wall; fail closed, which here
        # means changing nothing.
        for g, t, z, _f, _ in samples:
            out[(g, t, z)] = False
        return out
    for g in sorted(set(gid)):
        te = gid == g
        tr = ~te
        if not y[tr].any() or not (~y[tr]).any():
            for i in np.nonzero(te)[0]:
                out[(samples[i][0], samples[i][1], samples[i][2])] = False
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        A = (X[tr] - mu) / sd
        B = (X[te] - mu) / sd
        yt = y[tr]
        for k, i in enumerate(np.nonzero(te)[0]):
            d = np.linalg.norm(A - B[k], axis=1)
            dt = np.sort(d[yt])[:mnn].mean()
            dn = np.sort(d[~yt])[:mnn].mean()
            out[(samples[i][0], samples[i][1], samples[i][2])] = (
                (dn - dt) >= thr)
    return out
