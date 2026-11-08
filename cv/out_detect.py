"""Out-ordinal recogniser: which out of the inning is this, if any.

The last fact still read from the transcript is which column is which
inning, and the boundary it encodes is the third out. The recorder
draws the ordinal -- a Roman numeral -- and cv/README.md records the
sizing that pointed here: over the 31 sheets a cell whose batter is
out puts its clusters in BR and the CENTRE and almost nowhere else,
but only the centre is legible. Ten crude shape features with a fold
of one sheet reach 0.612 there against a 0.328 always-"none"
baseline, and 0.397 in BR against 0.386. This module is the same
chamfer template matching that reads the corner digits, pointed at the
centre; 0.612 is the floor it has to beat.

Classes: I / II / III (the ordinal), o (the run circle, which lives in
the same zone and must not be read as a numeral), and none.

The fold is one SHEET, never one cell -- the honest test pa_detect and
token_detect already use. One caveat is inherited from the corner
recogniser and stated rather than hidden: corner_feats computes its
chamfer SCALES over the whole pool, so a held-out sheet contributes to
the metric even though it can never be a template. That is the same
arrangement the published corner numbers are measured under.
"""
import os
import re

import annot_zones
import cell_diff
import corner_knn
import edge_score

MNN = max(1, int(os.environ.get("CORNER_OUT_MNN", "2")))
NONE_BIAS = float(os.environ.get("CORNER_OUT_NONE_BIAS", "0"))
ORDINAL = {1: "I", 2: "II", 3: "III"}


def label_of(truth):
    """GT label for the centre zone of one plate appearance."""
    out = truth.get("out") or {}
    if out.get("seq") in ORDINAL:
        return ORDINAL[out["seq"]]
    if truth.get("scored"):
        return "o"
    return "none"


def harvest(gt, aligned, template, x_play, pa=None):
    """[(tag, views, label)] for every plate appearance with a centre crop.

    pa: the tag set to read (the PA detector's, when it is on). A cell
    with no centre crop yields nothing at all -- there is no ink to
    judge, so calling it "none" for free would flatter the recogniser.

    Returns (samples, dropped). The second list is not decoration: a
    cell that CARRIES an ordinal and yields no crop is a miss the
    accuracy over the samples cannot see, and there are 119 of them
    against 392 that produce one. Report both or the number means the
    wrong thing.
    """
    tags = sorted(pa) if pa is not None else sorted(
        t for t, v in gt["cells"].items() if v.get("pa"))
    out, dropped = [], []
    for tag in tags:
        ri, ci = (int(x) for x in
                  re.fullmatch(r"r(\d+)c(\d+)", tag).groups())
        if (ri, ci) not in aligned:
            continue
        resid = cell_diff.residual(template, aligned[(ri, ci)])
        _, long_mask = edge_score.score_cell(resid, x_play)
        clusters, annot_mask = annot_zones.cell_clusters(
            resid, long_mask, x_play)
        lab = label_of(gt["cells"].get(tag, {}))
        views = corner_knn.zone_views(clusters, annot_mask, "center")
        if views is None:
            if lab not in ("none",):
                dropped.append((tag, lab))
            continue
        out.append((tag, views, lab))
    return out, dropped


def predict(samples):
    """samples: [(gid, tag, views, label)] -> {(gid, tag): label}.

    Leave-one-sheet-out: a sample is judged only against templates from
    OTHER sheets. Distance to a label is the mean of its MNN nearest
    templates, the same rule solve_corners uses for the corner digits.
    """
    feats = corner_knn.corner_feats([v for _, _, v, _ in samples])
    gids = [g for g, _, _, _ in samples]
    labs = [l for _, _, _, l in samples]
    pred = {}
    for i, (gid, tag, _, _) in enumerate(samples):
        pool = {}
        for j in range(len(samples)):
            if gids[j] == gid:
                continue
            pool.setdefault(labs[j], []).append(
                corner_knn.corner_dist(feats, i, j))
        if not pool:
            raise SystemExit(
                "out_detect: leave-one-sheet-out needs at least two "
                "sheets (got 1); use a games manifest")
        d = {lab: sum(sorted(ds)[:MNN]) / len(sorted(ds)[:MNN])
             for lab, ds in pool.items()}
        if "none" in d:
            d["none"] -= NONE_BIAS
        pred[(gid, tag)] = min(d, key=d.get)
    return pred
