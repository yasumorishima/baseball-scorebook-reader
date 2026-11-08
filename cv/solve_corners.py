#!/usr/bin/env python3
"""Constraint-solved corner numbers: recognition + game logic together.

A corner number (n) can only be a batting order that comes to the plate
AFTER this cell's batter within the same inning -- and within one cell
the advances happened on distinct, later and later plays, so the seq
positions behind (2B, 3B, home) strictly increase. Template matching
alone is starved at two games of data (33% top1); these constraints
shrink the label space from nine digits to the few batters who actually
batted behind the runner, which is where the missing accuracy lives.

Four predictors are compared against ground truth, all using the same
leave-one-out template pool:
  raw          -- nearest label, no constraints (corner_knn baseline)
  constrained  -- nearest label among valid later batters
  joint        -- per cell, the strictly-increasing assignment over the
                  present corners minimizing total distance
  occupancy    -- per inning, the cheapest combination of per-cell
                  assignments whose marks admit a LEGAL BASE-RUNNING
                  TRAJECTORY: one runner per base, no passing, walks are
                  dead balls advancing exactly the forced chain, and a
                  marked base is reached exactly at the marked play.

The occupancy model is deliberately loose everywhere the scorer is
loose (sound: it must never reject the true assignment):
  - unmarked advances happen silently on any play, bounded above by
    the runner's next marked base (arrival semantics);
  - runners with no marks left may silently leave the bases at any
    play (a run or an out the scorer did not annotate -- both occur
    in the reference game);
  - a PB or S corner is an arrival at an unspecified play (passed
    ball / steal), so those marks contribute NO agenda constraint;
  - WALKS ARE NOT DEAD-BALL-STRICT for runners. The first model
    allowed only the forced chain to move on a walk and rejected any
    other marked arrival at that play; the held-out game 3
    (2025-04-19) REFUTED that: its scorer marks mid-PA advances
    (wild pitch during the at-bat) with the batter's number, so
    r4c2 home:(8) arrives on a WALK play with first base empty.
    Walk plays now use live-play semantics with the batter placed on
    first; the force chain still emerges from occupancy (the batter
    displaces first base, ordering pushes the rest).

Combo-space history (all three designs measured -- read before touching):
  - v1 excluded PB from the per-cell combos after measuring that PB
    options dropped the two-game pooled occupancy below the joint
    baseline (an unconstrained label is a cheap escape hatch).
  - The game-3 held-out run then showed the OPPOSITE failure is worse
    locally: an S (steal) crop whose true label is time-unconstrained
    gets a numbered label FORCED onto it, and that wrong pinned event
    poisons the occupancy solution of the whole inning (003 inning 1:
    three cells dragged wrong; frozen-model occupancy 4/13 vs joint
    6/13 on the held-out game).
  - Opening S/PB everywhere, priced by distance, fixed game 3 (10/13)
    but collapsed the pool to 56%: cursive S/PB templates sit close to
    faint paren digits in soft-dice, so they siphoned pinned cells.
  - CURRENT RULE: an unconstrained label (S/PB) enters a corner's
    options only where recognition already ranks it top-1 for that
    crop. No tunable constant; recognition proposes, game logic
    disposes. S/PB picks still contribute no agenda marks. Known cost:
    an S that LOOKS like a digit (003 r1c0) still gets a number forced
    onto it and can distort its inning; known benefit: pinned numbered
    cells cannot be siphoned by a label that recognition itself does
    not favor. Numbers for all three designs: cv/README.md.

Staged inputs -- facts that came from the GT and were treated as
recognized, NOT as answer labels: which cells are plate appearances,
which corner zones carry a mark, and how each batter reached (walk/HBP
dead-ball force, triple 3B, HR home, anything else 1B; outs as batter
occupy nothing). The corner NUMBER was never taken from the GT: that is
the prediction.

ALL THREE now have recognisers, and each is switched on by an env flag
so every earlier measurement still reproduces:
  CORNER_MARK_PRED=1   which zones carry a mark      (2026-07-28)
  CORNER_REACH_PRED=1  out versus reached            (2026-07-27)
  CORNER_PA_PRED=1     which cells are plate appearances  (2026-08-30)
With all three on, nothing about the corner read comes from the GT
except the lattice bookkeeping -- which columns are innings and which
row leads off each inning -- and the corpus scores 295/356 = 83%
against 331/356 = 93% when the three facts are handed over. Numbers,
the operating-point sweep and the controls: cv/README.md.

Measured honestly (2026-07-04, after the game-3 held-out round trip):
the CURRENT model scores 37/42 on the two design sheets, 10/13 on the
held-out 2025-04-19 sheet, 45/55 = 82% pooled. The v1 model's 41/42 on
the design sheets was partly an artifact of the factually-wrong
dead-ball-walk assumption (game 3 refuted it; see below) -- four of
its pins do not survive the correction. The dominant remaining error
class is S/PB-vs-faint-digit template confusion, a recognition-layer
problem (per-sheet stroke normalization is the queued lever).

Note one deliberate asymmetry vs --selfcheck: the search's mark
presence comes from harvested CROPS (what annot_zones will supply in
production), so a true mark whose crop was absorbed into the trace
(002 r4c1 home) simply contributes no constraint -- weaker than the
GT's full mark set, never wrong.

Soft event-sharing prior: MEASURED AND REJECTED (2026-07-04, two games,
42 corners): SHARE 0.08 / 0.18 / 0.30 scored 62% / 55% / 50% against
the 64% per-cell joint baseline -- with template distances this noisy a
soft agreement bonus makes wrong events collapse into consensus. Do not
re-tune it. The occupancy predictor is the hard-constraint form of the
same structure: force pushes PIN events instead of rewarding agreement.

Usage: solve_corners.py <image> <grid.json> <gt.json> <ref_r,ref_c>
       solve_corners.py <games-manifest.json> [--selfcheck]
       (manifest pools every game's templates; constraints stay
        per-game. --selfcheck verifies the GT itself admits a legal
        trajectory in every inning -- run it after every GT or model
        edit: a failure means the model is too strict and will start
        rejecting true assignments.)
"""
import collections
import heapq
import io
import itertools
import os
import json
import re
import sys

import corner_knn
import token_detect

CORNER_ORDER = ["UR", "UL", "LL"]  # 2B, 3B, home -- advance order
ZONE_BASE = {"UR": 2, "UL": 3, "LL": 4}
LABEL_MNN = int(os.environ.get("LABEL_MNN", "2"))
# Per-label distance = mean of the label's LABEL_MNN nearest templates.
# 1 was the original rule (single nearest template) and is what every
# measurement before 2026-07-26 used; set LABEL_MNN=1 to reproduce them.
# Measured 2026-07-26 on 29 sheets / 346 corners, sweeping m = 1..5:
#   m=1  raw 186  con 228  joint 248  occupancy 284/346 (82%)
#   m=2  raw 204  con 243  joint 265  occupancy 295/346 (85%)  <- default
#   m=3  raw 210  con 245  joint 268  occupancy 293/346 (85%)
# Every stage improves, so this is not joint-assignment churn: raw is pure
# leave-one-out recognition with no solver involvement and it gains 18.
# Ledger vs m=1: 18 repaired / 7 broken, selfcheck PASS on all 29 games.
# Why it works: with a single nearest template one outlier can capture a
# sample -- the digit-siphoning S/PB templates were spread over ten
# different culprits, none dominant, so pruning was not available. Asking
# a label for two close templates instead of one removes the outlier's
# leverage. False S/PB calls fall 14 -> 10 and S/PB recall rises 25 -> 29.
# m is corpus-tuned but on a broad plateau (m=2 and m=3 both clearly beat
# m=1 at every stage); m=2 wins end-to-end and on false S/PB.
# The out-versus-reached fact is staged from GT, and it is the only
# staged quantity carrying real information: 444 outs against 369
# reaches, with extra-base hits accounting for 2 cells in the whole
# corpus. Before building a recogniser for it, it is worth knowing what
# accuracy that recogniser would have to hit. STAGE_FLIP inverts the
# fact on a deterministic 1-in-N slice of plate appearances so the cost
# per error can be read off directly. 0 disables.
STAGE_FLIP = int(os.environ.get("CORNER_STAGE_FLIP", "0"))
# Replace the staged out-versus-reached fact with the play-mark
# classifier measured 2026-07-27: the corner recogniser pointed at the
# BR + centre crop with a binary label, 84.4% leave-one-out over all
# 815 plate appearances. The staged budget predicts occupancy near 309
# at that accuracy; this makes the prediction real so the two can be
# compared. 0 keeps the GT fact and every earlier measurement.
REACH_PRED = os.environ.get("CORNER_REACH_PRED", "0") == "1"
REACH_MNN = int(os.environ.get("CORNER_REACH_MNN", "2"))
REACH_BIAS = float(os.environ.get("CORNER_REACH_BIAS", "0"))
# A mark in the UR zone is a runner advance to second base (see the
# notation model), and an out never carries an advance mark (0 of 860
# out plate appearances in GT, 2026-09-02). So any annotation ink in UR
# says the batter reached, whatever the play-mark crop looks like. Measured
# 2026-09-25 on 33 sheets: the play-mark kNN misses 104 of 419 reaches
# (46 of them have no BR/centre crop at all); OR-ing in "UR ink > 0"
# takes reach accuracy 791 -> 825 of 929 (recall 315 -> 367, false
# reaches 34 -> 52). The threshold is zero: nothing was tuned. 0 = off.
REACH_UR = os.environ.get("CORNER_REACH_UR", "0") == "1"
REACH_UR_INK = {}   # (gid, tag) -> annotation ink in UR, filled only when on
if REACH_UR and not REACH_PRED:
    print("warning: CORNER_REACH_UR=1 does nothing without CORNER_REACH_PRED=1",
          file=sys.stderr)
# Second staged fact: which zones carry an advance mark. Unlike reach,
# predicting it changes WHICH SAMPLES EXIST, so the denominator has to
# be pinned: it stays the set of GT-marked zones that produce a crop.
# A mark predicted absent is never read and scores wrong; a phantom
# mark joins the solve, contributing agenda constraints and noise, but
# is not a template and is not scored. 0 keeps the staged fact.
MARK_PRED = os.environ.get("CORNER_MARK_PRED", "0") == "1"
MARK_MNN = int(os.environ.get("CORNER_MARK_MNN", "2"))
MARK_BIAS = float(os.environ.get("CORNER_MARK_BIAS", "0"))
# A corner mark says THIS runner reached that base, so a cell whose
# reach fact says the runner never got on base cannot carry one.
# Measured 2026-09-02 on the ground truth: 0 of 860 plate
# appearances recorded as out carries an advance mark, while 21 of
# the detector's 81 accepted phantoms sit in exactly those cells.
# This is NOT the priced "not really a mark" option closed on
# 2026-08-31: nothing is discounted and the solver is given no
# choice. The corner is removed for contradicting a fact the same
# solver already consumes, so the two stages stop disagreeing.
MARK_REACH = os.environ.get("CORNER_MARK_REACH", "0") == "1"
# Diagnostic ceiling: drop every accepted corner the GT does not
# label. Reads the answer, so it can never ship -- it only bounds
# what a perfect verifier at this recall could be worth.
MARK_ORACLE = os.environ.get("CORNER_MARK_ORACLE", "0") == "1"
MARK_GT_REACH = os.environ.get("CORNER_MARK_GT_REACH", "0") == "1"
MARK_GT_SCORED = os.environ.get("CORNER_MARK_GT_SCORED", "0") == "1"
UNC_ORACLE = os.environ.get("CORNER_MARK_UNC_ORACLE", "0") == "1"
# The unnumbered tokens (WP, 1E/2E, DB, T.O, loose brackets) as one
# play-unknown class beside S and PB. Templates are the accepted
# corners the GT leaves unlabelled -- they are marks, they just
# carry no slot number.
UNC_CLASS = os.environ.get("CORNER_UNC_CLASS", "0") == "1"
UNC_LABEL = "UNC"
UNC_MARGIN = float(os.environ.get("CORNER_UNC_MARGIN", "0"))
# CORNER_TOKEN=1: recognise the corners that are marks WITHOUT a
# slot number (PB/S/WP/1E/DB/T.O and loose strokes) and give them an
# unconstrained label instead of letting them pin a batter. See
# cv/token_detect.py for why this is a separate recogniser and not
# another chamfer label class.
TOKEN_PRED = os.environ.get("CORNER_TOKEN", "0") == "1"
FORCE_UNC = set()                # (gid, tag, zone) decided by it
GT_RANK = {None: 0, "1B": 1, "2B": 2, "3B": 3, "home": 4}
# Third and last staged fact: which cells are plate appearances. This
# one decides which cells EXIST for the sequencer, so it is asymmetric
# in a way the other two are not: a missed plate appearance deletes its
# corners from the read AND deletes a legal label from every other
# cell's option list, while a phantom one only adds a sequence
# position, a candidate label and one more silent-advance opportunity.
# Denominator is pinned the same way as for marks -- it stays every
# GT-labelled corner that produces a crop, and a corner inside a cell
# the detector dropped is counted as never read. See cv/pa_detect.py.
PA_PRED = os.environ.get("CORNER_PA_PRED", "0") == "1"

# Fourth staged fact, and the first of the two lattice ledgers to go:
# which batting-order row leads off each inning. It is not read from the
# sheet at all -- an inning's batters are a contiguous run of the order,
# so the leadoff follows from the previous inning's last batter. Sized
# against the GT before this was built, over 174 innings on 31 sheets:
# the chain reproduces the GT ORDER in 143/143 innings that have a
# predecessor (the identity of the lead can differ where the inning is
# short enough that several leads sort the same, which is why the
# quantity to measure is the order, not the number), and every one of
# the 31 sheets opens with slot 1. So the ledger is worth 0 with the
# staged plate-appearance set; what it buys is insurance against a
# predicted PA set, where a missed or invented cell shifts the chain
# into every LATER inning.
# CORNER_LEAD_CHAIN=1 drops the ledger and chains; =2 also lets the
# inning's own slots speak: if exactly one start makes them a run of
# the order, that start wins, and the chain is only the fallback. On
# the staged PA set both are exact; on the predicted set the chain
# alone matches the ledger's ORDER in 160/174 innings and =2 in
# 165/174 (a phantom above the real leadoff drags the unique start
# back by the number of phantoms, which is the whole residual).
# DEFAULT 2 since 2026-09-03: the ledger is worth nothing on the
# staged set (331 either way, selfcheck clean) and three corners on
# the predicted one (331 -> 328, and 298 -> 295 with all three
# recognisers), so the shipped run stages one fewer fact for the same
# headline. CORNER_LEAD_CHAIN=0 puts the ledger back.
# Two things to close before =3 could ever be the default: --selfcheck
# validates the pre-pass order only, so it does not vouch for the order
# the solver actually picks; and an alternative order is not propagated,
# the next inning's chain and shortlist still come off the primary.
LEAD_CHAIN = int(os.environ.get("CORNER_LEAD_CHAIN", "2"))
# =3 stops deciding at all: both cues put a start forward, every
# distinct ORDER they name goes on a shortlist, and pass 2 pays for
# each and keeps the cheapest -- the corner evidence chooses instead
# of a blind pre-pass choosing for it. Sized first over the 31 sheets
# on the predicted PA set: the shortlist holds the ledger's own order
# in 171 of 174 innings, and 154 of them offer only one order, so the
# choice is real in 20 innings. CORNER_LEAD_ORACLE=1 reads the GT
# ledger to pick within the shortlist -- the diagnostic that separates
# "the shortlist covers it" from "the cost can find it".
# The last staged fact: which column is which inning. Sized on the GT
# first. A column is NOT a half-inning -- the recorder opens a new one
# when the batting order comes round onto cells already written, so
# sheet 002's second inning batted twelve and spilled slots 8 and 9
# into a fresh column, and sheet 001 spilled at the 11-slot wrap with
# only three rows used. The layout therefore cannot say where an
# inning ends; only the third out can. CORNER_COL_NAIVE=1 reads the
# columns that hold a plate appearance, in order, as innings 1..n --
# right about the PARTITION on 27 of the 31 sheets, wrong on exactly
# the four that spill (001, 002, 008, 009). It is a measuring stick
# for what the ledger is worth, not a replacement.
COL_NAIVE = os.environ.get("CORNER_COL_NAIVE", "0") == "1"
# CORNER_COL_SPLIT=1 replaces the ledger outright: cv/split_innings.py
# lays the sheet out as one batting-order sequence (recoverable from
# the columns alone -- verified against the ledger's order on all 31
# sheets) and cuts it on the out count, priced by the ordinal
# recogniser. Default off until the end-to-end number is measured.
COL_SPLIT = os.environ.get("CORNER_COL_SPLIT", "0") == "1"
RUNS_SEEN = [0, 0, 0, 0]
REACH_STATS = collections.Counter()
RUNS_STATS = collections.Counter()
SPLIT_DUMP = set(x for x in os.environ.get(
    "CORNER_SPLIT_DUMP", "").split(",") if x)
LEAD_ALT_CAP = 12
LEAD_ORACLE = os.environ.get("CORNER_LEAD_ORACLE", "0") == "1"
LEAD_STATS = {"alt": 0, "innings": 0}
# Estimate the chamfer scale factors from the LABELLED pool only: a
# sample that can never be a template should not set the scale that
# every template is measured against. Default ON since 2026-08-31 --
# with no phantoms in the pool it is bit-identical to the old rule
# (verified against the staged run), so no earlier measurement moves;
# CORNER_SCALE_LABELED=0 restores it. Ledger with all three facts
# recognised: 3 repaired, 0 broken, 295 -> 298.
# NOTE the hypothesis that motivated it was WRONG. It was built to undo
# the raw drop 269 -> 264 that the mark detector causes, on the theory
# that phantom crops move the metric. Per-corner comparison says
# otherwise: of that 5, two are corners deleted from the read outright
# and three are survivors damaged by the loss of five TEMPLATES from
# the pool -- and turning this flag on leaves raw at 264. The gain it
# does deliver lands in the constrained/joint/occupancy stages.
SCALE_LABELED = os.environ.get("CORNER_SCALE_LABELED", "1") == "1"
# Per-corner record of every stage's answer, for comparing two runs
# corner by corner instead of by their totals. Off unless a path is
# given; writing it changes nothing that is computed.
DUMP = os.environ.get("CORNER_DUMP", "")
NAMED_FROM = os.environ.get("CORNER_SPLIT_NAMED_FROM", "")
# Let a detected corner be dropped from its cell's assignment, priced by
# the mark detector's own margin (see the module note above). Needs
# CORNER_MARK_PRED=1: with the staged mark fact there are no phantoms to
# drop and no margin to price a drop with.
NOMARK = os.environ.get("CORNER_NOMARK", "0") == "1"
NOMARK_W = float(os.environ.get("CORNER_NOMARK_W", "1.0"))
NOMARK_FLOOR = float(os.environ.get("CORNER_NOMARK_FLOOR", "0"))
# Admit the drop option only where the detector was this unsure or
# less (its margin). Negative disables the gate = option everywhere,
# which is the priced form that measured 149-251/356.
NOMARK_GATE = float(os.environ.get("CORNER_NOMARK_GATE", "-1"))
# Admit the drop only where the corner fits its BEST label this
# badly or worse. 0 disables (no distance requirement).
NOMARK_DGATE = float(os.environ.get("CORNER_NOMARK_DGATE", "0"))
# Corners whose best label fits this badly or worse keep their
# place in the cell but may only take an UNCONSTRAINED label, so
# they cannot pin an event. 0 disables.
UNPIN_DGATE = float(os.environ.get("CORNER_UNPIN_DGATE", "0"))
NOMARK_LABEL = "NOMARK"          # sentinel: labels are digits or S/PB
# Claude's corner read as a proposal (experiments/claude_solver_2026-10,
# pre-registered in 8fd8c98). Off unless a tsv is given. With it, every
# label other than Claude's pays K, so the solver first minimises how many
# corners depart from Claude and only then uses the kNN distances; K is a
# lexicographic order, not a weight (the run asserts no inning can sum to
# K). SCOPE: both / cells (per-cell options only) / split (the inning
# split's named-batter read only). DROP leaves corners Claude gives no
# value out of the solve; they stay templates, and a GT one scores wrong.
CLAUDE_PATH = os.environ.get("CORNER_CLAUDE", "")
CLAUDE_K = float(os.environ.get("CORNER_CLAUDE_K", "1000"))
CLAUDE_SCOPE = os.environ.get("CORNER_CLAUDE_SCOPE", "both")
CLAUDE_DROP = os.environ.get("CORNER_CLAUDE_DROP", "0") == "1"
CLAUDE = {}                      # (gid, tag, zone) -> "7" / "S" / "PB"
if CLAUDE_PATH:
    assert CLAUDE_SCOPE in ("both", "cells", "split"), CLAUDE_SCOPE
    for _l in io.open(CLAUDE_PATH, encoding="utf-8"):
        if _l.strip():
            _k, _v = _l.rstrip("\n").split("\tclaude=")
            CLAUDE[tuple(_k.split(":"))] = _v
assert not CLAUDE_DROP or CLAUDE, "CORNER_CLAUDE_DROP needs CORNER_CLAUDE"
CLAUDE_STATS = collections.Counter()
CLAUDE_BASEMAX = {}              # sample -> its largest kNN label distance
CLAUDE_FB = set()                # (gid, inn) that fell back to joint
# Claude's read of the facts the logic stands on (experiments/
# claude_facts_2026-10): which cells are plate appearances, out versus
# reached (and triple / home run), and whether a cell carries an out
# numeral. Replaces the PA and reach recognisers and the split's ordinal
# prices; the corner marks and numbers are untouched. tsv lines:
# "gid:rXcY<TAB>pa=1<TAB>base=out|1|3|4<TAB>num=0|1".
CLAUDE_FACTS_PATH = os.environ.get("CORNER_CLAUDE_FACTS", "")
# This scorer writes BOTH corners when a runner takes two bases on one
# play -- (4) at third and (4) at home -- while the model required the
# corners of one cell to name strictly later and later plays, so a
# correct second mark was forced onto another slot. Of the 60 marks the
# GT leaves unlabelled that Claude reads (2026-10-14), 17 share their
# cell with a GT corner of the same slot. SAMEPLAY lets two corners of a
# cell name the same play, and the simulator then moves the runner
# through every base marked for that play at once. Per assignment it only
# loosens the model (the GT's strictly increasing marks stay feasible),
# but a whole inning can still end with no feasible pick under it (once
# on CONFIG B, 2026-10-14: sheet 009 inning 5, which the strict model
# solves; it also admits cells the strict model had no option for, a
# second way the same can happen). So an inning that finds nothing under
# SAMEPLAY is solved again under the strict model, without the cells only
# SAMEPLAY admitted, before it may fall back to the joint pick.
SAMEPLAY = os.environ.get("CORNER_CELL_SAMEPLAY", "0") == "1"
CLAUDE_FACTS = {}                # gid -> {tag: {"base": .., "num": ..}}
_seen_facts = set()
if CLAUDE_FACTS_PATH:
    assert not PA_PRED and not REACH_PRED and not STAGE_FLIP, (
        "CORNER_CLAUDE_FACTS replaces CORNER_PA_PRED, CORNER_REACH_PRED"
        " and CORNER_STAGE_FLIP")
    for _l in io.open(CLAUDE_FACTS_PATH, encoding="utf-8"):
        if not _l.strip():
            continue
        _k, *_f = _l.rstrip("\n").split("\t")
        _f = dict(x.split("=", 1) for x in _f)
        _g, _t = _k.split(":")
        assert _f["base"] in ("out", "1", "3", "4"), _l
        assert _f["pa"] in ("0", "1") and _f["num"] in ("0", "1"), _l
        assert (_g, _t) not in _seen_facts, _l
        _seen_facts.add((_g, _t))
        if _f["pa"] != "1":
            continue
        CLAUDE_FACTS.setdefault(_g, {})[_t] = {
            "base": None if _f["base"] == "out" else int(_f["base"]),
            "num": _f["num"] == "1"}
STATS = os.environ.get("CORNER_STATS", "0") == "1"
MARK_MARGIN = {}                 # (gid, tag, zone) -> detector margin
COMBO_KEEP = 4000     # covers (13 later + 1 gated S/PB)^3 = 2744; recheck
                      # if an inning ever exceeds ~15 plate appearances
OCC_CAP = 20000       # A* pops per inning before falling back to joint


def corner_cand_views(gt, aligned, template, x_play, pa=None, stats=None):
    """Every corner zone of every plate appearance that produces a crop,
    marked or not -- the candidate set a mark detector would see.

    pa: the cells to treat as plate appearances (the detector's set
    under CORNER_PA_PRED). A cell outside the GT carries no advances,
    so every zone it offers is an unlabelled phantom."""
    import annot_zones as _az
    import cell_diff
    import edge_score
    want = {"UR": "2B", "UL": "3B", "LL": "home"}
    out = {}
    tags = sorted(pa) if pa is not None else sorted(
        t for t, v in gt["cells"].items() if v.get("pa"))
    for tag in tags:
        truth = gt["cells"].get(tag, {})
        ri, ci = (int(x) for x in re.fullmatch(r"r(\d+)c(\d+)", tag).groups())
        resid = cell_diff.residual(template, aligned[(ri, ci)])
        _, long_mask = edge_score.score_cell(resid, x_play)
        clusters, annot_mask = _az.cell_clusters(resid, long_mask, x_play)
        adv = truth.get("advances", {})
        for zone, base in want.items():
            v = corner_knn.zone_views(clusters, annot_mask, zone)
            if v is None:
                continue
            out[(tag, zone)] = (v, str(adv[base]) if base in adv else None)
            if stats is not None:
                stats[(tag, zone)] = token_detect.zone_stats(clusters, zone)
    return out


def predict_marks(cand):
    """Leave-one-out mark/no-mark over every crop-producing corner zone."""
    keys = [(gid, tag, zone) for gid, mm in sorted(cand.items())
            for (tag, zone) in sorted(mm)]
    if not keys:
        return {}
    views = [cand[g][(t, z)][0] for g, t, z in keys]
    truth = [cand[g][(t, z)][1] is not None for g, t, z in keys]
    feats = corner_knn.corner_feats(views)
    n = len(keys)
    pred = {}
    for i in range(n):
        pool = {}
        for j in range(n):
            if j == i:
                continue
            pool.setdefault(truth[j], []).append(
                corner_knn.corner_dist(feats, i, j))
        best = {}
        for L, ds in pool.items():
            k = sorted(ds)[:MARK_MNN]
            best[L] = sum(k) / len(k)
        yes = best.get(True, 9.9) - MARK_BIAS
        no = best.get(False, 9.9)
        pred[keys[i]] = yes <= no
        # how much room the decision had: >= 0 exactly when accepted
        MARK_MARGIN[keys[i]] = no - yes
    return pred


def play_mark_views(gt, aligned, template, x_play, pa=None, ur=None):
    """BR + centre crop per plate appearance: the play notation plus the
    out numeral / run circle, which is where out-versus-reach is written."""
    import annot_zones as _az
    import cell_diff
    import edge_score
    out = {}
    tags = sorted(pa) if pa is not None else sorted(
        t for t, v in gt["cells"].items() if v.get("pa"))
    for tag in tags:
        ri, ci = (int(x) for x in re.fullmatch(r"r(\d+)c(\d+)", tag).groups())
        resid = cell_diff.residual(template, aligned[(ri, ci)])
        _, long_mask = edge_score.score_cell(resid, x_play)
        clusters, annot_mask = _az.cell_clusters(resid, long_mask, x_play)
        if ur is not None:
            ur[tag] = sum(c["ink"] for c in clusters if c["zone"] == "UR")
        boxes = [c["bbox"] for c in clusters if c["zone"] in ("BR", "center")]
        if not boxes:
            continue
        crop = corner_knn._crop_boxes(annot_mask, boxes)
        if crop is not None:
            out[tag] = [crop] * corner_knn.NVIEWS
    return out


def predict_reach(marks, structs):
    """Leave-one-out reach prediction over every play mark in the corpus.

    Returns {(gid, tag): True/False}. Cells with no play mark are absent
    and are treated as outs by the caller, which is the majority within
    that group."""
    keys = [(gid, tag) for gid, mm in sorted(marks.items())
            for tag in sorted(mm)]
    if not keys:
        return {}
    views = [marks[gid][tag] for gid, tag in keys]
    feats = corner_knn.corner_feats(views)
    # a phantom cell is not in the GT at all; production cannot know
    # that, so it enters the pool as what it looks like -- an out
    truth = [structs[gid][0]["cells"].get(tag, {}).get("reached") is not None
             for gid, tag in keys]
    n = len(keys)
    pred = {}
    for i in range(n):
        pool = {}
        for j in range(n):
            if j == i:
                continue
            pool.setdefault(truth[j], []).append(
                corner_knn.corner_dist(feats, i, j))
        best = {}
        for L, ds in pool.items():
            k = sorted(ds)[:REACH_MNN]
            best[L] = sum(k) / len(k)
        # BIAS > 0 leans toward calling a reach, which is the error the
        # solver absorbs better than a missed one.
        pred[keys[i]] = best.get(True, 9.9) - REACH_BIAS <= best.get(False, 9.9)
    return pred


def stage_facts(gt, pa=None):
    """tag -> {base} staged reach facts (see module docstring).

    Walk vs hit is deliberately NOT staged: game 3 refuted dead-ball
    strictness, so the simulator treats every play identically and
    forces emerge from occupancy alone.
    """
    facts = {}
    pas = sorted(pa) if pa is not None else sorted(
        t for t, v in gt["cells"].items() if v.get("pa"))
    flip_at = set()
    if STAGE_FLIP:
        step = max(1, round(100.0 / STAGE_FLIP))
        flip_at = set(pas[::step])
    for tag in pas:
        truth = gt["cells"].get(tag, {})
        via = truth.get("reach_via") or ""
        if truth.get("reached") is None:
            facts[tag] = {"base": None}
        elif via.startswith("HR"):
            facts[tag] = {"base": 4}
        elif via.startswith("triple"):
            facts[tag] = {"base": 3}
        else:
            facts[tag] = {"base": 1}
        if tag in flip_at:
            facts[tag] = {"base": None if facts[tag]["base"] else 1}
    return facts


def inning_sequences(gt, pa=None, alts=None,
                     ledger=False):
    """inning -> [(tag, slot), ...] in batting (temporal) order.

    slot = batting-order number the cell belongs to; usually the row,
    but an 11-batter lineup overflows slot 11 into row 10 of the next
    column (batter_slot in the GT). Row-wrap sorting still orders that
    correctly because the overflow cell sits in a later column.

    A continuation column may hold BOTH tail-of-first-pass slots and
    second-pass slots (game 9: c6 = r1,r2,r3 first pass + r4,r5 second
    pass). Positions are computed from the SLOT (so the 11-slot
    overflow cell keeps its place between slot 10 and slot 1), modulo
    L = 11 when an 11th slot exists, else 10; a cell whose position
    does not exceed the previous column's maximum is lifted by +L --
    within one column first-pass tails always outrank second-pass
    heads, so the per-column comparison is sound. Pure sequencing
    plumbing: the baseball constraints in feasible() are untouched,
    and games 1-8 order bit-identically under this rule (verified).
    """
    lat = gt["lattice"]
    col_inn = {int(c.lstrip("c")): inn
               for c, inn in lat["column_to_inning"].items()}
    mode = 0 if ledger else LEAD_CHAIN
    leadoff = {} if mode else {
        int(i): int(r.lstrip("r"))
        for i, r in lat["inning_leadoff_row"].items()}
    pa_rc = {}
    tags = sorted(pa) if pa is not None else sorted(
        t for t, v in gt["cells"].items() if v.get("pa"))
    for tag in tags:
        ri, ci = (int(x) for x in
                  re.fullmatch(r"r(\d+)c(\d+)", tag).groups())
        # a phantom cell has no GT entry, so its slot is its row --
        # which is what an 11-slot overflow sheet would get wrong, and
        # is the honest production default. Latent hazard, not fired by
        # any sheet in the corpus (all have ten data rows): a phantom in
        # an eleventh row would make L below 11 for the WHOLE sheet and
        # shift every wrap position.
        pa_rc[(ri, ci)] = (tag, gt["cells"].get(tag, {}).get(
            "batter_slot", ri))
    if COL_NAIVE and not ledger:
        # not under ledger=True: that call exists to read the ledger,
        # and its leadoff numbers are the ledger's own, so naive
        # columns there would make the oracle disagree with itself on
        # exactly the sheets whose columns spill
        _cs = sorted({_ci for _ri, _ci in pa_rc})
        col_inn = {_c: _k + 1 for _k, _c in enumerate(_cs)}
    L = 11 if any(s >= 11 for _, s in pa_rc.values()) else 10
    if mode >= 2:
        # the run test needs the sheet's real lineup length, not a
        # fixed 10: a nine-batter sheet read modulo 10 shows a hole at
        # slot 10 in every inning that wraps, and no start matches.
        # The deepest slot in use is it. Trying to be cleverer than
        # that cost a selfcheck: "deepest slot used at least twice"
        # was meant to ignore a lone phantom in an unused row, and on
        # sheet 009 -- a real ten-batter lineup whose tenth slot bats
        # exactly once -- it read L as 9, left inning 4 with no valid
        # start, and made the GT's own marks infeasible.
        # a sheet with no plate appearance at all keeps the fixed
        # modulus: max() of nothing raises, and the pre-refactor
        # line (11-or-10) could not. Not reachable on this corpus.
        if pa_rc:
            L = max(_s for _, _s in pa_rc.values())
    seqs = {}
    chain_prev = None       # slot of the last batter of the last inning
    for inn in sorted(set(col_inn.values())):
        cols = sorted(c for c, i in col_inn.items() if i == inn)
        chlead, cands = None, []
        if mode:
            chlead = 1 if chain_prev is None else chain_prev % L + 1
            lead = chlead
            if mode >= 2:
                ss = [sl for (r, c), (t, sl) in pa_rc.items()
                      if c in cols]
                for s0 in range(1, L + 1):
                    run, cur = [], s0
                    for _ in range(len(ss)):
                        run.append(cur)
                        cur = cur % L + 1
                    if sorted(run) == sorted(ss):
                        cands.append(s0)
                if len(cands) == 1:
                    lead = cands[0]
        else:
            lead = leadoff[inn]

        def ordered(lead, cols=cols):
            seq, prev_max = [], -1
            for c in cols:
                keyed = []
                for (r, cc), (tag, slot) in pa_rc.items():
                    if cc != c:
                        continue
                    pos = (slot - lead) % L
                    if pos <= prev_max:
                        pos += L
                    keyed.append((pos, r))
                keyed.sort()
                if keyed:
                    prev_max = keyed[-1][0]
                seq.extend(pa_rc[(r, c)] for _, r in keyed)
            return seq

        seq = ordered(lead)
        seqs[inn] = seq
        if alts is not None and mode >= 3:
            out = [seq]
            seen_o = {tuple(t for t, _ in seq)}
            for l2 in ([chlead] if chlead else []) + cands:
                if len(out) >= LEAD_ALT_CAP:
                    break
                s2 = ordered(l2)
                k2 = tuple(t for t, _ in s2)
                if k2 not in seen_o:
                    seen_o.add(k2)
                    out.append(s2)
            alts[inn] = out
        if seq:
            chain_prev = seq[-1][1]
        # an inning with no plate appearance (the side did not bat)
        # carries the chain across untouched
    return seqs, col_inn


def feasible(seq_facts, agendas):
    """Does a legal trajectory exist matching the marked arrivals?

    seq_facts: per seq position {base: 1/3/4/None}.
    agendas: seq position p -> ((base, q_or_None), ...) base-ascending;
    q None = PB/S (arrival play unknown; the search never emits these,
    only --selfcheck does). Configs are tuples of active runners
    (p, base, agenda_index), p ascending -- no-passing therefore reads
    as strictly DESCENDING bases along the tuple.
    """
    def valid(runners):
        return all(runners[i][1] > runners[i + 1][1]
                   for i in range(len(runners) - 1))

    def pb_expand(cfgs):
        # consume any subset of pending PB marks, in any legal order
        out, frontier = set(cfgs), list(cfgs)
        while frontier:
            cfg = frontier.pop()
            for i, (p, b, ai) in enumerate(cfg):
                a = agendas.get(p, ())
                if ai < len(a) and a[ai][1] is None and a[ai][0] > b:
                    nb = a[ai][0]
                    if nb >= 4:
                        if ai + 1 != len(a):
                            continue
                        nxt = cfg[:i] + cfg[i + 1:]
                    else:
                        nxt = cfg[:i] + ((p, nb, ai + 1),) + cfg[i + 1:]
                        if not valid(nxt):
                            continue
                    if nxt not in out:
                        out.add(nxt)
                        frontier.append(nxt)
        return out

    def live_step(cfg, q, fact):
        per_runner = []
        for (p, b, ai) in cfg:
            a = agendas.get(p, ())
            opts = []
            if ai < len(a) and a[ai][1] == q:      # mandatory arrival
                j = ai
                while SAMEPLAY and j + 1 < len(a) and a[j + 1][1] == q:
                    j += 1                         # two bases, one play
                nb = a[j][0]
                if nb > b:
                    opts.append(None if nb >= 4 else (p, nb, j + 1))
            else:
                bound = a[ai][0] if ai < len(a) else 5
                opts.extend((p, nb, ai) for nb in range(b, min(bound, 4)))
                if ai == len(a):
                    opts.append(None)              # unmarked run or out
            if not opts:
                return set()
            per_runner.append(opts)
        if fact["base"] is not None and fact["base"] < 4:
            ba = agendas.get(q, ())
            if ba and ba[0][0] <= fact["base"]:
                return set()                       # own mark not above start
        results = set()
        for choice in itertools.product(*per_runner):
            runners = [c for c in choice if c is not None]
            if fact["base"] is not None:
                if fact["base"] >= 4:              # HR sweeps the bases
                    if runners:
                        continue
                else:
                    runners.append((q, fact["base"], 0))
            if valid(runners):
                results.add(tuple(runners))
        return results

    cfgs = {()}
    for q, fact in enumerate(seq_facts):
        cfgs = pb_expand(cfgs)
        nxt = set()
        for cfg in cfgs:
            nxt |= live_step(cfg, q, fact)
        if not nxt:
            return False
        cfgs = nxt
    cfgs = pb_expand(cfgs)                         # PB during the last play
    return any(all(ai == len(agendas.get(p, ())) for p, _, ai in cfg)
               for cfg in cfgs)


def build_agendas(group, picks, pos_in_seq, inn):
    """group cells + chosen combo index per cell -> runner agendas."""
    agendas = {}
    for cell, k in zip(group, picks):
        if k is None:
            continue
        p = pos_in_seq[(inn, cell["tag"])]
        marks = tuple((ZONE_BASE[z], q) for z, (q, _)
                      in zip(cell["present"], cell["combos"][k][1])
                      if q is not None)   # PB/S carry no time constraint
        if marks:
            agendas[p] = marks                     # base-ascending already
    return agendas


OCC_STATS = {"none": 0, "ok": 0}


def occupancy_pick(group, seq_facts, pos_in_seq, inn):
    """Cheapest feasible joint choice of per-cell combos, via A*.

    Cells are assigned one at a time; a prefix's marks are a RELAXATION
    of any completion (unassigned cells contribute no marks, and marks
    only tighten), so an infeasible prefix prunes its whole subtree.
    That monotonicity holds because build_agendas emits numbered marks
    only -- PB/S picks (q=None) are dropped there, so even though they
    sit in the combo space they never add pb_expand trajectories to a
    search agenda and the prefix pruning stays sound.
    The heuristic (cheapest combo of every unassigned cell) is
    admissible, so the first fully-assigned feasible node popped is the
    optimal feasible assignment.
    """
    group = sorted(group, key=lambda c: pos_in_seq[(inn, c["tag"])])
    lists = [c["combos"] for c in group]
    tail = [0.0] * (len(lists) + 1)
    for i in range(len(lists) - 1, -1, -1):
        tail[i] = tail[i + 1] + lists[i][0][0]
    heap = [(tail[0], ())]
    pops = 0
    while heap and pops < OCC_CAP:
        f, prefix = heapq.heappop(heap)
        pops += 1
        agendas = build_agendas(group, prefix, pos_in_seq, inn)
        if not feasible(seq_facts, agendas):
            continue
        i = len(prefix)
        if i == len(lists):
            OCC_STATS["ok"] += 1
            return group, prefix, pops
        g = f - tail[i]
        for k, (cost, _) in enumerate(lists[i]):
            heapq.heappush(heap, (g + cost + tail[i + 1], prefix + (k,)))
    OCC_STATS["none"] += 1                         # no feasible assignment
    return group, None, pops                       # fall back to joint


def selfcheck(games_structs):
    """The GT itself must be feasible in every inning; else the model
    is too strict and will reject true assignments in the search."""
    bad = 0
    for gid, (gt, seqs, _tag_inn, pos_in_seq, facts) in games_structs.items():
        for inn, seq in seqs.items():
            seq_facts = [facts[tag] for tag, _ in seq]
            slot_pos = {}
            for k, (_, slot) in enumerate(seq):
                slot_pos.setdefault(str(slot), []).append(k)
            per_runner = []
            for k, (tag, _) in enumerate(seq):
                # under CORNER_PA_PRED a sequenced cell can be a phantom
                # with no GT entry at all; it carries no marks
                adv = gt["cells"].get(tag, {}).get("advances") or {}
                marks = []
                for bname, bnum in (("2B", 2), ("3B", 3), ("home", 4)):
                    if bname not in adv:
                        continue
                    v = adv[bname]
                    if v in ("PB", "S"):           # play unknown
                        marks.append([(bnum, None)])
                    else:
                        marks.append([(bnum, q) for q
                                      in slot_pos.get(str(v), []) if q > k])
                if marks:
                    per_runner.append((k, marks))
            ok = False
            for combo in itertools.product(
                    *(itertools.product(*m) for _, m in per_runner)):
                agendas = {p: tuple(c) for (p, _), c
                           in zip(per_runner, combo)}
                if all(all(a[i][1] is None or a[i + 1][1] is None or
                           a[i][1] < a[i + 1][1]
                           for i in range(len(a) - 1))
                       for a in agendas.values()) and \
                        feasible(seq_facts, agendas):
                    ok = True
                    break
            if not per_runner:
                ok = feasible(seq_facts, {})
            status = "ok" if ok else "INFEASIBLE"
            if not ok:
                bad += 1
            print(f"  selfcheck {gid} inning {inn}: {status}")
    print("selfcheck:", "PASS" if bad == 0 else f"FAIL ({bad} innings)")
    return bad == 0


def predict_pa(games):
    """gid -> set of tags the detector calls plate appearances.

    Costs one extra pass over the sheets (the residuals are cheap and
    are thrown away again), which keeps peak memory at one sheet.
    """
    import cell_diff
    import pa_detect
    samples = []
    for g in games:
        with open(g["gt"]) as f:
            gt = json.load(f)
        with open(g["grid"]) as f:
            grid = json.load(f)
        cols, rows = cell_diff.lattice(grid)
        strip_w = cell_diff.strip_width(grid["v_clusters"])
        ref_rc = tuple(int(x) for x in g["ref"].split(","))
        aligned, template, x_play = corner_knn.load_game(
            g["image"], g["grid"], ref_rc)
        inn_cols = {int(c.lstrip("c"))
                    for c in gt["lattice"]["column_to_inning"]}
        feats = pa_detect.sheet_features(
            aligned, template, x_play, strip_w, rows, cols, inn_cols)
        truth = {t for t, v in gt["cells"].items() if v.get("pa")}
        for tag, f in sorted(feats.items()):
            samples.append((g["id"], tag, f, tag in truth))
    pred = pa_detect.predict(samples)
    out = {g["id"]: set() for g in games}
    tp = fn = fp = 0
    for (gid, tag, _, real) in samples:
        got = pred[(gid, tag)]
        if got:
            out[gid].add(tag)
        tp += got and real
        fn += (not got) and real
        fp += got and (not real)
    print("PA detector: %d/%d real kept (recall %.4f), %d missed, "
          "%d phantom over %d candidate cells"
          % (tp, tp + fn, tp / float(tp + fn), fn, fp, len(samples)))
    return out


def cell_combos(gid, tag, zones, dists, present, later,
                count=None, sameplay=None):
    """Priced options for one cell under ONE ordering of the
    inning.

    Split out of pass 1 because the options name LATER batters:
    change the leadoff and the same cell is a different pricing
    problem, so a shortlisted order has to be rebuilt, not
    re-sorted. Nothing else in a cell depends on the order.
    """
    count = {True: 0, False: 0} if count is None else count
    opts = []
    for z in present:
        ld = dists[z]
        rawlab = min(ld, key=ld.get) if ld else None
        if TOKEN_PRED and (gid, tag, z) in FORCE_UNC:
            opts.append([(None, "PB")])
            continue
        if (UNC_CLASS and rawlab == UNC_LABEL and UNC_MARGIN > 0):
            _oth = {L: d for L, d in ld.items() if L != UNC_LABEL}
            if _oth and min(_oth.values()) - ld[UNC_LABEL] < UNC_MARGIN:
                rawlab = min(_oth, key=_oth.get)
        if UNC_CLASS and rawlab == UNC_LABEL:
            count[zones[z][1] is not None] += 1
        if UNC_ORACLE and zones[z][1] is None:
            opts.append([(None, "PB")])
            continue
        extra = ([(None, rawlab)]
                 if rawlab in ("PB", "S", UNC_LABEL) else [])
        if NOMARK and (gid, tag, z) in MARK_MARGIN and (
                NOMARK_GATE < 0
                or MARK_MARGIN[(gid, tag, z)] <= NOMARK_GATE) and (
                NOMARK_DGATE <= 0
                or (ld and min(ld.values()) >= NOMARK_DGATE)):
            extra = extra + [(None, NOMARK_LABEL)]
        if (UNPIN_DGATE > 0 and ld
                and min(ld.values()) >= UNPIN_DGATE):
            # keeps its cost, loses its power to pin a play
            unc = [(None, L) for L in ("PB", "S") if L in ld]
            opts.append(unc or later + extra)
        else:
            opts.append(later + extra)
    combos = []
    for combo in itertools.product(*opts):
        qs = [q for q, _ in combo if q is not None]
        sp = SAMEPLAY if sameplay is None else sameplay
        if any(qs[k] > qs[k + 1] if sp else qs[k] >= qs[k + 1]
               for k in range(len(qs) - 1)):
            continue
        base = 0.0
        for z, (_, lab) in zip(present, combo):
            if lab == NOMARK_LABEL:
                # the detector's own confidence is the price
                base += NOMARK_FLOOR + NOMARK_W * max(
                    0.0, MARK_MARGIN.get((gid, tag, z), 0.0))
            else:
                base += dists[z].get(lab, 3.0)  # unseen: high, not inf
        combos.append((base, combo))
    combos.sort(key=lambda t: t[0])
    valid = ({lab for _, lab in later} | {"PB", "S"}
             | ({UNC_LABEL} if UNC_CLASS else set()))
    if len(combos) > COMBO_KEEP:
        CLAUDE_STATS["combo_cap"] += 1
    return combos[:COMBO_KEEP], valid


def main():
    argv = [a for a in sys.argv[1:] if a != "--selfcheck"]
    do_selfcheck = "--selfcheck" in sys.argv
    if len(argv) >= 4:       # single game, legacy CLI
        games = [{"id": "g0", "image": argv[0], "grid": argv[1],
                  "gt": argv[2], "ref": argv[3]}]
    else:                    # games manifest (see corner_knn docstring)
        with open(argv[0]) as f:
            games = json.load(f)

    # pooled template set; per-game constraint structures
    ss = []          # (crop, label, gid, tag, zone)
    marks = {}       # gid -> {tag: play-mark views}, only when REACH_PRED
    cand = {}        # gid -> {(tag,zone): (views, label)}, only when MARK_PRED
    cand_stats = {}  # gid -> {(tag,zone): per-zone cluster stats}
    missed = 0       # GT marks the detector never surfaced
    structs = {}     # gid -> (gt, seqs, tag_inn, pos_in_seq, facts)
    alt_seqs = {}    # gid -> {inn: [seq, ...]} shortlist (CORNER_LEAD_CHAIN=3)
    ledger_seqs = {}  # gid -> {inn: seq} (CORNER_LEAD_ORACLE diagnostic only)
    pa_pred = predict_pa(games) if PA_PRED else None
    for g in games:
        with open(g["gt"]) as f:
            gt = json.load(f)
        gt_pa = {t for t, v in gt["cells"].items() if v.get("pa")}
        pa = pa_pred[g["id"]] if PA_PRED else gt_pa
        if CLAUDE_FACTS_PATH:
            pa = set(CLAUDE_FACTS.get(g["id"], {}))
        dropped = gt_pa - pa
        ref_rc = tuple(int(x) for x in g["ref"].split(","))
        aligned, template, x_play = corner_knn.load_game(
            g["image"], g["grid"], ref_rc)
        for crop, lab, src_tag in corner_knn.harvest(
                gt, aligned, template, x_play)["corner"]:
            tag, zone = src_tag.split(":")
            if tag in dropped:
                missed += 1      # a corner inside a cell never read
                continue
            ss.append((crop, lab, g["id"], tag, zone))
        alts = {}
        seqs, col_inn = inning_sequences(gt, pa, alts)
        alt_seqs[g["id"]] = alts
        if LEAD_ORACLE:
            ledger_seqs[g["id"]] = inning_sequences(
                gt, pa, ledger=True)[0]
        pos_in_seq = {}
        tag_inn = {}
        for inn, seq in seqs.items():
            for k, (tag, _) in enumerate(seq):
                pos_in_seq[(inn, tag)] = k
                tag_inn[tag] = inn
        structs[g["id"]] = (gt, seqs, tag_inn, pos_in_seq,
                            stage_facts(gt, pa))
        if REACH_PRED:
            ur = {} if REACH_UR else None
            marks[g["id"]] = play_mark_views(gt, aligned, template, x_play,
                                             pa, ur)
            for t, v in (ur or {}).items():
                REACH_UR_INK[(g["id"], t)] = v
        if MARK_PRED:
            cand_stats[g["id"]] = {}
            cand[g["id"]] = corner_cand_views(gt, aligned, template, x_play,
                                              pa, cand_stats[g["id"]])

    if REACH_PRED:
        pr = predict_reach(marks, structs)
        hit = tot = 0
        for gid, (gt, seqs, _tag_inn, pos_in_seq, facts) in structs.items():
            for tag in list(facts):          # the PA set actually solved
                got = pr.get((gid, tag), False)
                if REACH_UR and REACH_UR_INK.get((gid, tag), 0) > 0:
                    got = True
                facts[tag] = {"base": 1 if got else None}
                truth = gt["cells"].get(tag, {})
                if not truth.get("pa"):
                    continue                 # phantom: solved, not scored
                tot += 1
                hit += got == (truth.get("reached") is not None)
        print("reach predictor: %d/%d = %.1f%% of real plate appearances"
              % (hit, tot, 100.0 * hit / tot))

    if CLAUDE_FACTS_PATH:
        _ids = {g["id"] for g in games}
        assert set(CLAUDE_FACTS) <= _ids, sorted(set(CLAUDE_FACTS) - _ids)
        assert all(CLAUDE_FACTS.get(i) for i in _ids), sorted(
            i for i in _ids if not CLAUDE_FACTS.get(i))
        _st = collections.Counter()
        for gid, (gt, _s, _ti, _p, facts) in structs.items():
            gpa = {t for t, v in gt["cells"].items() if v.get("pa")}
            _st["pa_both"] += len(gpa & set(facts))
            _st["pa_missed"] += len(gpa - set(facts))
            _st["pa_phantom"] += len(set(facts) - gpa)
            for tag in list(facts):
                facts[tag] = {"base": CLAUDE_FACTS[gid][tag]["base"]}
                if tag in gpa:
                    want = stage_facts(gt, {tag})[tag]["base"]
                    _st["reach_tot"] += 1
                    _st["reach_ok"] += ((facts[tag]["base"] is None)
                                        == (want is None))
                    _st["base_ok"] += facts[tag]["base"] == want
                    if want is not None:
                        _st["num_tot"] += 1
                        _st["num_ok"] += (CLAUDE_FACTS[gid][tag]["num"]
                                          == ("out" in gt["cells"][tag]))
        print("claude facts: PA %d both, %d missed, %d phantom; reach %d/%d, "
              "base %d/%d; out numeral on reached cells %d/%d"
              % (_st["pa_both"], _st["pa_missed"], _st["pa_phantom"],
                 _st["reach_ok"], _st["reach_tot"], _st["base_ok"],
                 _st["reach_tot"], _st["num_ok"], _st["num_tot"]))

    if MARK_PRED:
        mp = predict_marks(cand)
        if MARK_GT_REACH or MARK_GT_SCORED:
            ngr = nglost = 0
            for gid, mm in sorted(cand.items()):
                cells = structs[gid][0]["cells"]
                for (tag, zone), (_v, lab) in sorted(mm.items()):
                    if not mp.get((gid, tag, zone), False):
                        continue
                    t = cells.get(tag, {})
                    r = GT_RANK[t.get("reached")]
                    drop = False
                    if MARK_GT_REACH and ZONE_BASE[zone] > r:
                        drop = True
                    if (MARK_GT_SCORED and zone == "LL"
                            and not t.get("scored")):
                        drop = True
                    if drop:
                        mp[(gid, tag, zone)] = False
                        ngr += 1
                        nglost += lab is not None
            print("mark GT gate: %d accepted corners dropped, %d of them "
                  "GT marks" % (ngr, nglost))
        if MARK_ORACLE:
            n_or = 0
            for gid, mm in sorted(cand.items()):
                for (tag, zone), (_v, lab) in sorted(mm.items()):
                    if lab is None and mp.get((gid, tag, zone), False):
                        mp[(gid, tag, zone)] = False
                        n_or += 1
            print("mark ORACLE: %d unlabelled corners dropped" % n_or)
        if MARK_REACH:
            nrej = nlost = 0
            for gid, mm in sorted(cand.items()):
                facts = structs[gid][4]
                for (tag, zone), (_v, lab) in sorted(mm.items()):
                    if not mp.get((gid, tag, zone), False):
                        continue
                    if facts.get(tag, {}).get("base") is None:
                        mp[(gid, tag, zone)] = False
                        nrej += 1
                        nlost += lab is not None
            print("mark/reach consistency: %d accepted corners dropped, "
                  "%d of them GT marks" % (nrej, nlost))
        ss = []
        tp = fp = 0
        mmissed = 0      # display only: `missed` is shared with the PA
                         # stage, and a corner lost upstream is not the
                         # mark detector's recall to answer for
        for gid, mm in sorted(cand.items()):
            for (tag, zone), (views, lab) in sorted(mm.items()):
                if not mp.get((gid, tag, zone), False):
                    if lab is not None:
                        missed += 1
                        mmissed += 1
                    continue
                if lab is not None:
                    tp += 1
                else:
                    fp += 1
                ss.append((views, lab, gid, tag, zone))
        print("mark detector: %d real kept, %d missed, %d phantom "
              "(recall %.3f)"
              % (tp, mmissed, fp, tp / float(tp + mmissed)))
        if STATS:
            # Can a GATE on the detector's own margin separate the
            # phantoms from the real marks? If the two distributions sit
            # on top of each other, no threshold can, and the gated form
            # is dead before it is run.
            real_m, ph_m = [], []
            for gid, mm in sorted(cand.items()):
                for (tag, zone), (_, lab) in sorted(mm.items()):
                    k = (gid, tag, zone)
                    if not mp.get(k, False) or k not in MARK_MARGIN:
                        continue
                    (real_m if lab is not None else ph_m).append(
                        MARK_MARGIN[k])
            for g in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
                nr = sum(1 for m in real_m if m <= g)
                np_ = sum(1 for m in ph_m if m <= g)
                print("  gate<=%.2f exposes %3d/%3d phantom (%.0f%%) and "
                      "risks %3d/%3d real (%.0f%%)"
                      % (g, np_, len(ph_m), 100.0 * np_ / max(1, len(ph_m)),
                         nr, len(real_m), 100.0 * nr / max(1, len(real_m))))

    if TOKEN_PRED and MARK_PRED:
        samples = []
        for gid, mm in sorted(cand.items()):
            for (tag, zone), (views, lab) in sorted(mm.items()):
                if not mp.get((gid, tag, zone), False):
                    continue
                zs = cand_stats.get(gid, {}).get((tag, zone), {})
                samples.append((
                    gid, tag, zone,
                    token_detect.feats(
                        views[0], zs, zone,
                        MARK_MARGIN.get((gid, tag, zone), 0.0)),
                    lab is None))
        tp_ = token_detect.predict(samples)
        adm = sip = 0
        for (gid, tag, zone, _f, is_tok) in samples:
            if tp_.get((gid, tag, zone), False):
                FORCE_UNC.add((gid, tag, zone))
                adm += is_tok
                sip += not is_tok
        print("token detector: %d unnumbered marks unpinned, %d GT marks "
              "siphoned (of %d token / %d numbered accepted)"
              % (adm, sip, sum(1 for s in samples if s[4]),
                 sum(1 for s in samples if not s[4])))

    if do_selfcheck and not selfcheck(structs):
        sys.exit(1)

    feats = corner_knn.corner_feats(
        [c for c, _, _, _, _ in ss],
        scale_mask=([lab is not None for _, lab, _, _, _ in ss]
                    if SCALE_LABELED else None))

    def label_dist(i):
        """label -> LOO distance for sample i, averaged over the label's
        LABEL_MNN nearest templates (LABEL_MNN=1 is the historical
        nearest-template rule and is bit-identical to it)."""
        if LABEL_MNN <= 1:
            best = {}
            for j, (_, lab, _, _, _) in enumerate(ss):
                if j == i:
                    continue
                d = corner_knn.corner_dist(feats, i, j)
                if d < best.get(lab, 9.9):
                    best[lab] = d
            return best
        pool = {}
        for j, (_, lab, _, _, _) in enumerate(ss):
            if j == i:
                continue
            if lab is None:
                if not UNC_CLASS:
                    continue           # phantom marks are not templates
                lab = UNC_LABEL
            pool.setdefault(lab, []).append(corner_knn.corner_dist(feats, i, j))
        return {lab: sum(sorted(ds)[:LABEL_MNN]) / len(sorted(ds)[:LABEL_MNN])
                for lab, ds in pool.items()}

    def claude_dist(i, site):
        """label_dist with Claude's proposal applied at this call site."""
        ld = label_dist(i)
        if not CLAUDE or not ld:
            return ld
        CLAUDE_BASEMAX[i] = max(ld.values())
        L = CLAUDE.get(tuple(ss[i][2:5]))
        if L is None or CLAUDE_SCOPE not in ("both", site):
            return ld
        assert L in ld, ("no template for Claude label", ss[i][2:5], L)
        CLAUDE_STATS["applied_" + site] += 1
        return {X: d if X == L else d + CLAUDE_K for X, d in ld.items()}

    # group samples by cell so joint assignment sees all its corners
    by_cell = {}
    for i, (_, lab, gid, tag, zone) in enumerate(ss):
        if CLAUDE_DROP and (gid, tag, zone) not in CLAUDE:
            CLAUDE_STATS["dropped"] += 1
            CLAUDE_STATS["dropped_gt"] += lab is not None
            continue
        by_cell.setdefault((gid, tag), {})[zone] = (i, lab)

    if COL_SPLIT:
        # Replace the ledger's column->inning map with a partition cut
        # from the sheet itself. Placed here, after the raw corner
        # labels exist, so the cut can be priced by what the recogniser
        # says the marks NAME -- the constraint that does the pruning --
        # without reading the GT's answer for it. Costs one more pass
        # over the images for the centre crops, like predict_pa.
        import out_detect
        import split_innings
        _samples = []
        for g in games:
            if CLAUDE_FACTS_PATH:
                break               # the numeral comes from Claude instead
            _gt = structs[g["id"]][0]
            _pa = set(structs[g["id"]][4])
            _ref = tuple(int(x) for x in g["ref"].split(","))
            _al, _tm, _xp = corner_knn.load_game(g["image"], g["grid"], _ref)
            _got, _ = out_detect.harvest(_gt, _al, _tm, _xp, _pa)
            for _tag, _views, _lab in _got:
                _samples.append((g["id"], _tag, _views, _lab))
        if CLAUDE_FACTS_PATH:
            # numeral_price's sign convention: negative = a numeral is there
            _prices = {(gid, tag): (-1.0 if v["num"] else 1.0)
                       for gid, d in CLAUDE_FACTS.items()
                       for tag, v in d.items()}
        else:
            _prices = split_innings.prices_leave_one_sheet_out(_samples)
        _named = {}
        for (gid, tag), zones in sorted(by_cell.items()):
            got = []
            for zone, (i, _lab) in sorted(zones.items()):
                ld = claude_dist(i, "split")
                if not ld:
                    continue
                r = min(ld, key=ld.get)
                if r.isdigit():
                    got.append(r)
                    if (CLAUDE and CLAUDE_SCOPE in ("both", "split")
                            and CLAUDE.get(tuple(ss[i][2:5])) == r):
                        CLAUDE_STATS["named_from_claude"] += 1
            if got:
                _named[(gid, tag)] = got
        if NAMED_FROM:
            # Re-cut with reads taken from a run whose partition is not
            # the one being tested. cell_combos only ever offers batters
            # from the SAME segment, so reads produced under partition A
            # agree with A by construction and cannot be used to judge a
            # cut derived from them. Reads from the naive left-to-right
            # map are not built that way, so they can.
            _path, _col = NAMED_FROM.rsplit(":", 1)
            _named = {}
            for _line in io.open(_path, encoding="utf-8"):
                _line = _line.rstrip(chr(10))
                if not _line:
                    continue
                _key, _rest = _line.split("	", 1)
                _gid, _tag, _zone = _key.split(":")
                _fields = dict(p.split("=", 1) for p in _rest.split("	"))
                _v = _fields.get(_col, "")
                if _v.isdigit():
                    _named.setdefault((_gid, _tag), []).append((_zone, _v))
            _named = {k: [v for _, v in sorted(vs)]
                      for k, vs in _named.items()}
            print("named-batter term re-fed from %s column %s: %d cells"
                  % (_path, _col, len(_named)))
        kept = same = cut_hit = cut_tot = 0
        for gid, (gt, seqs, tag_inn, pos_in_seq, facts) in list(
                structs.items()):
            flat = split_innings.flat_order(gt, set(facts))
            was = [tuple(t for t, _ in seqs[i]) for i in sorted(seqs)]
            # Is the TRUTH even in the candidate set? Count the
            # ledger's own innings that this run could have
            # chosen: at most OUT_CAP predicted batter outs
            # and playable. A cut cannot find an inning it
            # is not allowed to propose.
            for _inn, _seq in seqs.items():
                _b = sum(1 for _t, _ in _seq
                         if facts.get(_t, {}).get("base") is None)
                REACH_STATS["innings"] += 1
                REACH_STATS["outs_ok"] += (
                    _b <= split_innings.OUT_CAP)
                if _b <= split_innings.OUT_CAP and (
                        split_innings.inning_playable(
                            gt, facts, _seq, feasible)):
                    REACH_STATS["admissible"] += 1
                REACH_STATS["outs3"] += (_b == 3)
            new_seqs = split_innings.split(
                gt, facts, flat,
                {t: p for (g_, t), p in _prices.items() if g_ == gid},
                feasible,
                {t: v for (g_, t), v in _named.items() if g_ == gid},
                gt.get("runs_by_inning") or (
                    gt.get("reconstruction_checks") or {}
                ).get("runs_per_inning"))
            if new_seqs is None:
                kept += 1               # nothing playable: keep the ledger
                continue
            now = [tuple(t for t, _ in new_seqs[i]) for i in sorted(new_seqs)]
            _rs = gt.get("runs_by_inning") or (
                gt.get("reconstruction_checks") or {}).get("runs_per_inning")
            if _rs:
                RUNS_SEEN[0] += 1
                RUNS_SEEN[1] += len(_rs)
                for _nm, _ss in (("ledger", was), ("chosen", now)):
                    RUNS_SEEN[2 if _nm == "ledger" else 3] += len(_ss)
                    for _k, _seg in enumerate(_ss):
                        _R = _rs.get(str(_k + 1))
                        if _R is None:
                            RUNS_STATS[_nm + "_noentry"] += 1
                            continue
                        _lob = len(_seg) - 3 - _R
                        _lo, _hi = (-3, 3) if _k == len(_ss) - 1 else (0, 3)
                        if _lo <= _lob <= _hi:
                            RUNS_STATS[_nm + "_in"] += 1
                        else:
                            RUNS_STATS[_nm + "_out"] += 1
                            if _nm == "chosen" and _seg not in was:
                                RUNS_STATS["chosen_out_and_wrong"] += 1
                for _seg in now:
                    if _seg not in was:
                        RUNS_STATS["chosen_wrong"] += 1
            if gid in SPLIT_DUMP:
                _sl = {t: s_ for t, s_ in flat}
                print("  split dump %s: flat = %s" % (gid, " ".join(
                    "%s/%s" % (t, _sl[t]) for t, _ in flat)))
                for _nm, _ss in (("ledger", was), ("chosen", now)):
                    print("    %s: %s" % (_nm, " | ".join(
                        " ".join("%s/%s" % (t, _sl.get(t, "?")) for t in seg)
                        for seg in _ss)))
            same += was == now
            cut_tot += len(was)
            cut_hit += len(set(was) & set(now))
            pos_in_seq, tag_inn = {}, {}
            for inn, seq in new_seqs.items():
                for k, (tag, _) in enumerate(seq):
                    pos_in_seq[(inn, tag)] = k
                    tag_inn[tag] = inn
            structs[gid] = (gt, new_seqs, tag_inn, pos_in_seq, facts)
        print("inning split: %d/%d sheets identical to the ledger, "
              "%d/%d innings identical, %d sheets fell back"
              % (same, len(structs), cut_hit, cut_tot, kept))
        print("  header line score, on %d sheets (%d innings recorded): "
              "ledger %d in / %d out, chosen %d in / %d out; "
              "of %d chosen innings the ledger does not have, %d break the "
              "bound; segments ledger %d chosen %d, entries %d"
              % (RUNS_SEEN[0], RUNS_SEEN[1], RUNS_STATS["ledger_in"],
                 RUNS_STATS["ledger_out"], RUNS_STATS["chosen_in"],
                 RUNS_STATS["chosen_out"], RUNS_STATS["chosen_wrong"],
                 RUNS_STATS["chosen_out_and_wrong"], RUNS_SEEN[2],
                 RUNS_SEEN[3], RUNS_SEEN[1]))
        _st = split_innings.STATS
        print("  price terms that actually paid, over %d chosen segments: "
              "trailing %d, named-batter %d (%d violations), ordinal %d"
              % (_st["segments"], _st["trail"], _st["named"],
                 _st["violations"], _st["priced"]))
        print("  segments that needed a fourth batter out: %d"
              % _st["overout"])
        _rs2 = REACH_STATS
        print("  the ledger's own innings, judged by THIS run's facts: "
              "%d/%d within the out cap, %d also playable, %d have "
              "exactly three predicted batter outs"
              % (_rs2["outs_ok"], _rs2["innings"],
                 _rs2["admissible"], _rs2["outs3"]))
        print("  one-sided named count as a segment grows by one cell: "
              "%d steps up, %d down, %d level (it is not monotone)"
              % (_st["mono_up"], _st["mono_down"], _st["mono_same"]))
        if do_selfcheck:
            # The selfcheck above ran at :1067, BEFORE this block replaced
            # structs, so it judged the LEDGER partition. That is the check
            # it exists for -- is the GT itself playable -- but a run with
            # this flag on was reported as "selfcheck passes under the
            # predicted partition", which it never measured. Ask the real
            # question here, and label it.
            print("selfcheck of the PREDICTED partition "
                  "(the one above judged the ledger):")
            if not selfcheck(structs):
                print("  the cut the model chose cannot play every inning "
                      "of the GT's own marks. Not fatal to the corner score "
                      "-- the solver drops what it cannot place -- but it "
                      "is a defect in the partition, and the split's own "
                      "feasibility test cannot catch it because at "
                      "CORNER_SPLIT_MARKS=0 it never reads the marks.")

    # ---- pass 1: per-cell candidate combos ----
    # each present corner takes a later batter's event (numbered qs
    # strictly increasing within the cell) or a play-unknown PB/S
    # (priced by distance, no agenda constraint: see docstring)
    cells = []
    _unc_top1 = {True: 0, False: 0}
    for (gid, tag), zones in sorted(by_cell.items()):
        _, seqs, tag_inn, pos_in_seq, _ = structs[gid]
        inn = tag_inn[tag]
        seq = seqs[inn]
        p = pos_in_seq[(inn, tag)]
        later = [(q, str(slot)) for q, (_, slot) in enumerate(seq) if q > p]
        dists = {z: claude_dist(i, "cells") for z, (i, _) in zones.items()}
        present = [z for z in CORNER_ORDER if z in zones]
        combos, valid = cell_combos(
            gid, tag, zones, dists, present, later,
            _unc_top1)
        cells.append({"gid": gid, "inn": inn, "tag": tag, "zones": zones,
                      "present": present, "dists": dists,
                      "valid": valid,
                      "combos": combos,
                      "pick": 0 if combos else None})

    if CLAUDE:
        _bound = collections.Counter()
        for c in cells:
            for z in c["present"]:
                _bound[(c["gid"], c["inn"])] += CLAUDE_BASEMAX.get(
                    c["zones"][z][0], 0.0)
        CLAUDE_STATS["bound_x1000"] = int(1000 * max(_bound.values()))
        assert max(_bound.values()) < CLAUDE_K, max(_bound.values())

    if UNC_CLASS:
        print("UNC top-1: %d unlabelled corners (of the pool), "
              "%d GT marks siphoned"
              % (_unc_top1[False], _unc_top1[True]))
    if STATS:
        def _pct(vals, name):
            v = sorted(vals)
            if not v:
                print("  %s: none" % name)
                return
            q = [v[min(len(v) - 1, int(f * len(v)))]
                 for f in (0, .05, .25, .5, .75, .95)]
            print("  %-22s n=%4d  %s"
                  % (name, len(v), " ".join("%.3f" % x for x in q)))
        print("distributions (min, 5, 25, 50, 75, 95 pct):")
        _pct(list(MARK_MARGIN.values()), "mark-detector margin")
        _pct([min(c["dists"][z].values()) for c in cells
              for z in c["present"] if c["dists"][z]], "best label distance")
        _pct([c["dists"][z][lab] for c in cells for z in c["present"]
              if c["zones"][z][1] in c["dists"][z]
              for lab in [c["zones"][z][1]]], "TRUE label distance")
        _pct([min(c["dists"][z].values()) for c in cells
              for z in c["present"]
              if c["dists"][z] and c["zones"][z][1] is not None],
             "best dist: REAL corner")
        _pct([min(c["dists"][z].values()) for c in cells
              for z in c["present"]
              if c["dists"][z] and c["zones"][z][1] is None],
             "best dist: PHANTOM")
        _pct([len(c["present"]) for c in cells], "corners per cell")

    # ---- pass 2: per-inning occupancy search ----
    by_inning = {}
    for c in cells:
        if c["combos"]:
            by_inning.setdefault((c["gid"], c["inn"]), []).append(c)
    for (gid, inn), group in sorted(by_inning.items()):
        _, seqs, _, _, facts = structs[gid]
        shortlist = alt_seqs.get(gid, {}).get(inn) or [seqs[inn]]
        LEAD_STATS["innings"] += 1
        keep = dict(OCC_STATS)   # an inning counts once, not
                                # once per order tried
        best = None                   # per shortlisted order
        for ai, cseq in enumerate(shortlist):
            pis = {(inn, t): k for k, (t, _) in enumerate(cseq)}
            if ai:
                trial = []
                for c in group:
                    p = pis[(inn, c["tag"])]
                    later = [(q, str(sl)) for q, (_, sl)
                             in enumerate(cseq) if q > p]
                    trial.append(cell_combos(
                        c["gid"], c["tag"], c["zones"], c["dists"],
                        c["present"], later))
            else:
                trial = primary = [(c["combos"], c["valid"])
                                   for c in group]
            for c, (cb, vl) in zip(group, trial):
                c["combos"], c["valid"] = cb, vl
            if not all(c["combos"] for c in group):
                continue              # an order that prices nothing
            ordered, idx, pops = occupancy_pick(
                group, [facts[t] for t, _ in cseq], pis, inn)
            if CLAUDE:
                CLAUDE_STATS["maxpops"] = max(CLAUDE_STATS["maxpops"], pops)
                CLAUDE_STATS["capped"] += pops >= OCC_CAP
            if idx is None:
                continue
            cost = sum(c["combos"][k][0] for c, k in zip(ordered, idx))
            if LEAD_ORACLE:
                # diagnostic: take the ledger's own order when the
                # shortlist holds it, so a shortfall can be read as
                # "the shortlist missed it" or "the cost missed it"
                ref = tuple(t for t, _ in ledger_seqs[gid][inn])
                cost = 0.0 if tuple(t for t, _ in cseq) == ref else 1e6
            if best is None or cost < best[0] - 1e-9:
                best = (cost, ai, trial, list(zip(ordered, idx)))
        OCC_STATS.clear()
        OCC_STATS.update(keep)
        if best is None and SAMEPLAY:
            cseq = shortlist[0]
            pis = {(inn, t): k for k, (t, _) in enumerate(cseq)}
            strict = []
            for c in group:
                p = pis[(inn, c["tag"])]
                later = [(q, str(sl)) for q, (_, sl)
                         in enumerate(cseq) if q > p]
                strict.append(cell_combos(
                    c["gid"], c["tag"], c["zones"], c["dists"],
                    c["present"], later, sameplay=False))
            sub = [(c, t) for c, t in zip(group, strict) if t[0]]
            if sub:
                for c, (cb, vl) in sub:
                    c["combos"], c["valid"] = cb, vl
                ordered, idx, pops = occupancy_pick(
                    [c for c, _ in sub], [facts[t] for t, _ in cseq],
                    pis, inn)
                if idx is not None:
                    group = [c for c, _ in sub]
                    best = (0.0, 0, [t for _, t in sub],
                            list(zip(ordered, idx)))
                    primary = [t for _, t in sub]
                    CLAUDE_STATS["strict_retry"] += 1
                    print(f"  occupancy {gid} inning {inn}: no feasible "
                          f"pick under SAMEPLAY, solved strict without "
                          f"{len(strict) - len(sub)} SAMEPLAY-only cells")
        if best is None:
            # nothing was feasible: put the primary order's pricing
            # back, so the joint fallback is read off the order the
            # pre-pass actually chose and not the last one tried
            for c, (cb, vl) in zip(group, primary):
                c["combos"], c["valid"] = cb, vl
            OCC_STATS["none"] += 1
            CLAUDE_FB.add((gid, inn))
            print(f"  occupancy {gid} inning {inn}: no feasible pick "
                  f"in any of {len(shortlist)} orders, keeping joint")
        else:
            OCC_STATS["ok"] += 1
            for c, (cb, vl) in zip(group, best[2]):
                c["combos"], c["valid"] = cb, vl
            for cell, k in best[3]:
                cell["pick"] = k
            LEAD_STATS["alt"] += best[1] != 0

    def claude_fields(cell, z):
        if not CLAUDE:
            return ""
        cl = CLAUDE.get((cell["gid"], cell["tag"], z))
        return "\tcl=%s\tclvalid=%d\tfb=%d" % (
            cl if cl is not None else "-",
            cl is not None and cl in cell["valid"],
            (cell["gid"], cell["inn"]) in CLAUDE_FB)

    # ---- scoring ----
    n = raw_ok = con_ok = joint_ok = occ_ok = 0
    dump = []
    for cell in cells:
        joint_pred = {z: lab for z, (_, lab) in zip(
            cell["present"],
            cell["combos"][0][1] if cell["combos"] else [])}
        occ_pred = {z: lab for z, (_, lab) in zip(
            cell["present"],
            cell["combos"][cell["pick"]][1]
            if cell["pick"] is not None else [])}
        for z, (i, lab) in cell["zones"].items():
            if lab is None:                # phantom: solved, never scored
                if DUMP:
                    dump.append("%s:%s:%s	true=PHANTOM	raw=-	con=-	"
                                "joint=%s	occ=%s"
                                % (cell["gid"], cell["tag"], z,
                                   joint_pred.get(z), occ_pred.get(z))
                                + claude_fields(cell, z))
                continue
            n += 1
            ld = cell["dists"][z]
            raw = min(ld, key=ld.get) if ld else None
            cvalid = {k: v for k, v in ld.items() if k in cell["valid"]}
            con = min(cvalid, key=cvalid.get) if cvalid else None
            raw_ok += raw == lab
            con_ok += con == lab
            joint_ok += joint_pred.get(z) == lab
            occ_ok += occ_pred.get(z) == lab
            if DUMP:
                dump.append("%s:%s:%s\ttrue=%s\traw=%s\tcon=%s\t"
                            "joint=%s\tocc=%s"
                            % (cell["gid"], cell["tag"], z, lab, raw, con,
                               joint_pred.get(z), occ_pred.get(z))
                            + claude_fields(cell, z))
            if occ_pred.get(z) != lab:
                print(f"  {cell['gid']}:{cell['tag']}:{z} true={lab} "
                      f"raw={raw} con={con} joint={joint_pred.get(z)} "
                      f"occ={occ_pred.get(z)}")
    if DUMP:
        with open(DUMP, "w") as f:
            f.write("\n".join(sorted(dump)) + "\n")
        print("dumped %d scored corners to %s" % (len(dump), DUMP))
    n += missed          # a mark never surfaced is a corner never read
    n += CLAUDE_STATS["dropped_gt"]   # left out by CORNER_CLAUDE_DROP
    if SAMEPLAY or CLAUDE or CLAUDE_FACTS_PATH:
        print("sameplay strict retries %d, cells at COMBO_KEEP %d"
              % (CLAUDE_STATS["strict_retry"], CLAUDE_STATS["combo_cap"]))
    if CLAUDE:
        print("claude proposals: %d values, scope %s, K %g: applied cells %d "
              "split %d, named from claude %d, dropped %d (%d GT), "
              "max inning distance sum %.3f, max A* pops %d, capped %d, "
              "fallback innings %d %s"
              % (len(CLAUDE), CLAUDE_SCOPE, CLAUDE_K,
                 CLAUDE_STATS["applied_cells"], CLAUDE_STATS["applied_split"],
                 CLAUDE_STATS["named_from_claude"], CLAUDE_STATS["dropped"],
                 CLAUDE_STATS["dropped_gt"],
                 CLAUDE_STATS["bound_x1000"] / 1000.0, CLAUDE_STATS["maxpops"],
                 CLAUDE_STATS["capped"], len(CLAUDE_FB), sorted(CLAUDE_FB)))
    if LEAD_CHAIN >= 3:
        print("leadoff shortlist: %d innings, %d took an order other "
              "than the pre-pass pick" % (LEAD_STATS["innings"],
                                          LEAD_STATS["alt"]))
    print("occupancy A*: %d innings solved, %d with no feasible "
          "assignment" % (OCC_STATS["ok"], OCC_STATS["none"]))
    print(f"corner digits (n={n}): raw {raw_ok}/{n} = {raw_ok / n:.0%}  "
          f"constrained {con_ok}/{n} = {con_ok / n:.0%}  "
          f"joint {joint_ok}/{n} = {joint_ok / n:.0%}  "
          f"occupancy {occ_ok}/{n} = {occ_ok / n:.0%}")


if __name__ == "__main__":
    main()
