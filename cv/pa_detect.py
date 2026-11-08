#!/usr/bin/env python3
"""Plate-appearance detection: the third and last staged fact.

Until now solve_corners was told which cells are plate appearances. That
fact decides which cells enter the batting sequence at all, so replacing
it is not the same kind of substitution as the mark and reach
predictors: a MISSED plate appearance deletes its corners from the read
(they can never be scored right) and also deletes a legal label from
every other cell's option list, while a PHANTOM one adds a sequence
position, an extra candidate label, and one more play on which runners
may silently move. Both hurt, but the miss hurts categorically and the
phantom only dilutes -- so the operating point is deliberately pushed to
the recall side, exactly as the 2026-07-28 mark-detector measurement
required.

What the cell shows: a batter who came to the plate leaves ink in the
count strip (the pitch sequence), in the play area (the notation), or
both, and usually a base trace. A cell nobody batted in is blank. The
confounders are the inked cells that are NOT plate appearances -- a
substitution zigzag with a name written through it, a wild-pitch brace,
an inning-end double bar -- and they are the whole difficulty: over the
31-sheet corpus 452 of the 940 non-PA cells in inning columns carry
20 or more residual ink pixels.

Features are per cell -- twelve numbers, all from the residual after the
median template is subtracted: ink in the play area, ink gathered into
annot_zones clusters (total, largest, count, and per zone: five zones),
the long-stroke mask edge_score finds (a base trace), and the
count-strip marks (count and area). Classifier
is the same leave-one-out nearest-neighbour rule the other two staged
facts use -- mean distance to the m nearest samples of each class, minus
a bias that leans toward calling a plate appearance -- but the fold is a
whole SHEET, not a single cell: cells from one sheet share a hand, a
pen, an exposure and a scorer's habits, so a cell-level fold has a
near-duplicate of the held-out cell available to it. Whether that
actually flatters the score was MEASURED, not assumed,
and it does not: CORNER_PA_LOO=cell gives the identical 857 kept / 45
phantom at the default operating point and is marginally WORSE at bias
0.0 (847/27 against 848/26). The sheet fold stays because it is the
honest test, not because it costs anything -- which also says the
thirteen features are not sheet-idiosyncratic.

Not replaced here, and named so it is not quietly claimed: the lattice
itself (grid_detect), which columns are innings and which row leads off
each inning (GT lattice). This module answers only "did somebody bat in
this cell".
"""
import os

import numpy as np

import annot_zones
import cell_diff
import count_strip
import edge_score

ZONES = ["UL", "UR", "LL", "BR", "center"]
LOG_FEATS = ["raw_ink", "clus_ink", "max_clus", "long_px", "strip_area"]
LIN_FEATS = ["n_clus", "strip_n"]


def cell_features(resid, x_play, strip_w):
    """Everything the detector is allowed to see about one cell."""
    h, w = resid.shape
    pad = cell_diff.PAD
    binr = resid >= edge_score.INK_ON
    interior = np.zeros_like(binr, bool)
    interior[pad + 2:-(pad + 2), pad + 2:-(pad + 2)] = True
    play = np.zeros_like(binr, bool)
    play[:, x_play:] = True
    raw_ink = int(np.count_nonzero(binr & interior & play))

    _, long_mask = edge_score.score_cell(resid, x_play)
    clusters, _ = annot_zones.cell_clusters(resid, long_mask, x_play)
    zink = {z: 0 for z in ZONES}
    for c in clusters:
        zink[c["zone"]] = zink.get(c["zone"], 0) + int(c["ink"])
    inks = sorted((int(c["ink"]) for c in clusters), reverse=True)
    marks = count_strip.strip_marks(resid, strip_w)
    return {
        "raw_ink": raw_ink,
        "long_px": int(np.count_nonzero(long_mask > 0)),
        "n_clus": len(clusters),
        "clus_ink": int(sum(inks)),
        "max_clus": inks[0] if inks else 0,
        "strip_n": len(marks),
        "strip_area": int(sum(m["area"] for m in marks)),
        "zink": zink,
    }


def sheet_features(aligned, template, x_play, strip_w, rows, cols, inn_cols):
    """tag -> features for every data cell of every inning column."""
    out = {}
    for ri in range(1, len(rows)):
        for ci in range(len(cols)):
            if ci not in inn_cols:
                continue
            resid = cell_diff.residual(template, aligned[(ri, ci)])
            out["r%dc%d" % (ri, ci)] = cell_features(resid, x_play, strip_w)
    return out


def _matrix(feats):
    cols = [np.log1p([f[k] for f in feats]) for k in LOG_FEATS]
    cols += [np.array([float(f[k]) for f in feats]) for k in LIN_FEATS]
    cols += [np.log1p([f["zink"].get(z, 0) for f in feats]) for z in ZONES]
    return np.stack(cols, 1).astype(np.float64)


def predict(samples, mnn=None, bias=None, fold=None,
            margins=None):
    """samples: [(gid, tag, features, is_pa)] -> {(gid, tag): bool}.

    Standardisation is refit inside every fold, so nothing about the
    held-out sheet -- not even its spread -- reaches the rule that
    judges it.
    """
    mnn = int(os.environ.get("CORNER_PA_MNN", "5")) if mnn is None else mnn
    bias = (float(os.environ.get("CORNER_PA_BIAS", "1.0"))
            if bias is None else bias)
    fold = os.environ.get("CORNER_PA_LOO", "sheet") if fold is None else fold
    # Control the classifier has to beat: call every cell with any ink a
    # plate appearance. It is not a straw man -- over the corpus it
    # reaches recall 0.9977, one cell BETTER than the classifier, and if
    # phantoms really were free it would be the right answer. Its cost
    # is 639 phantoms against 45, and cv/README.md records what that
    # buys end to end.
    if os.environ.get("CORNER_PA_RULE", "knn") == "ink":
        return {(g, t): f["raw_ink"] > 0 for g, t, f, _ in samples}
    X0 = _matrix([f for _, _, f, _ in samples])
    y = np.array([p for _, _, _, p in samples], bool)
    gid = np.array([g for g, _, _, _ in samples])
    n = len(samples)
    pred = np.zeros(n, bool)
    if fold == "sheet" and len(set(gid)) < 2:
        # Every row would be held out at once: the training slice is
        # empty, its mean and std are nan, and nan <= nan is False, so
        # the detector would silently call NOTHING a plate appearance.
        # Reproduced on one sheet: 60 cells in, 0 kept, 22 real plate
        # appearances lost, no warning. The single-game CLI reaches
        # here.
        raise SystemExit(
            "pa_detect: CORNER_PA_LOO=sheet needs at least two sheets "
            "(got %d); use a games manifest, or CORNER_PA_LOO=cell"
            % len(set(gid)))
    folds = ([(gid == g) for g in sorted(set(gid))] if fold == "sheet"
             else [np.arange(n) == i for i in range(n)])
    for held in folds:
        tr = ~held
        mu, sd = X0[tr].mean(0), X0[tr].std(0)
        sd[sd == 0] = 1.0
        X = (X0 - mu) / sd
        d2 = ((X[held][:, None, :] - X[tr][None, :, :]) ** 2).sum(-1)
        D = np.sqrt(d2)
        ytr = y[tr]
        dp = np.sort(D[:, ytr], 1)[:, :mnn].mean(1)
        dn = np.sort(D[:, ~ytr], 1)[:, :mnn].mean(1)
        pred[held] = dp - bias <= dn
        if margins is not None:
            # how far inside the accept side the cell sits; the same
            # quantity the rule thresholds, reported not re-derived
            for i, m in zip(np.nonzero(held)[0], dn - (dp - bias)):
                margins[(samples[i][0], samples[i][1])] = float(m)
    return {(g, t): bool(p) for (g, t, _, _), p in zip(samples, pred)}
