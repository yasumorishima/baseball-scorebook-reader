"""Cut the sheet into innings without reading which column is which.

`column_to_inning` is the last fact the solver still takes from the
transcript, and cv/README.md records why the obvious replacements do
not work: a new column starts when the batting order laps back onto
cells already written, not when an inning ends, so the boundary is not
on the page; the naive left-to-right rule makes the GT itself
infeasible on four innings; and `//` is drawn 12 times in 175 innings.

What IS on the page is the out. The ordinal generator built from it
demanded each inning's numerals be exactly {1,2,3}, which is a gate:
one misread deletes the sheet, and fed the recogniser's own output it
kept the truth on 3 sheets of 31. Widening the gate was measured and
rejected (candidates 15,157, median 84 after feasibility). So the gate
became a price, and then the model was rebuilt around what the corpus
actually says:

  * every one of the 398 batter-out cells carries an ordinal, so
    `reached is None` already names most of the numerals and the
    recogniser adds nothing there;
  * a closed inning has exactly three outs, so the cells whose batters
    reached can only be outs by being RUNNER outs -- 27 of them are;
  * 140 closed innings of 143 end AT their last batter-out, the other
    three being innings whose third out was a runner's.

So a segment is admissible when it holds at most three batter outs and
the occupancy test can play it, and it is priced by two terms compared
lexicographically -- no exchange rate to invent:

  1. cells trailing after the last batter out (0 for 140 of 143)
  2. what the ordinal recogniser charges for the runner outs the count
     still needs, its own distance between the numeral classes and
     {none, o}

Measured over the 31 sheets, staged reach, leave-one-sheet-out for the
recogniser: the truth is reachable on 31 sheets (the gate managed 3),
is priced cheapest on 26, and an arbitrary tie-break inside the
cheapest band puts 159 of 174 inning boundaries in the right place --
the best member of that band would get 168. Term 1 alone gets 130 and
term 2 alone 141, so both are load-bearing.

Not claimed: this is measured with the reach facts STAGED. Nothing here
is fitted -- no threshold is chosen and no constant is tuned -- but the
lexicographic order of the two terms was picked after seeing both.
"""
import bisect
import collections
import itertools
import os
import re

BIG = 1e6
INF = (BIG, BIG, BIG)
NUMERALS = ("I", "II", "III")
NONE_LIKE = ("none", "o")
COMBO_CAP = int(os.environ.get("CORNER_SPLIT_COMBO_CAP", "50000"))
# How much of the corner marks the occupancy test may see.
#   0  nothing: the segment is judged on the reach facts alone
#   1  PRESENCE only -- which corners carry a mark, not which batter
#      they name. This is the same staging the published pipeline
#      already runs under: it harvests the corner crops from the GT and
#      predicts their VALUES, so presence is not extra knowledge, and
#      an unvalued mark enters the test exactly as PB/S does.
#   2  the GT values too. That is the answer the solver is trying to
#      read, so a number measured at 2 is not a result -- diagnostic.
USE_MARKS = int(os.environ.get("CORNER_SPLIT_MARKS", "0"))


def flat_order(gt, pa=None):
    """The whole sheet as one batting-order sequence: [(tag, slot)].

    The lineup does not reset at an inning boundary, so a sheet is a
    single cyclic run of slots 1..L and each column holds a contiguous
    stretch of it. Chain the columns left to right, starting each one
    where the previous ended. Verified to reproduce the order the GT
    lattice produces on all 31 sheets; on 5 columns the chained start
    is not one of the contiguous starts (a gap in the run) and the
    first contiguous start is used instead, which agreed every time.
    """
    tags = sorted(pa) if pa is not None else sorted(
        t for t, v in gt["cells"].items() if v.get("pa"))
    pa_rc = {}
    for tag in tags:
        ri, ci = (int(x) for x in
                  re.fullmatch(r"r(\d+)c(\d+)", tag).groups())
        pa_rc[(ri, ci)] = (tag, gt["cells"].get(tag, {}).get(
            "batter_slot", ri))
    if not pa_rc:
        return []
    L = max(s for _, s in pa_rc.values())
    out, prev = [], None
    for c in sorted({ci for _, ci in pa_rc}):
        col = [(r, t, s) for (r, cc), (t, s) in pa_rc.items() if cc == c]
        if not col:
            continue
        want = 1 if prev is None else prev % L + 1
        starts = []
        for s0 in range(1, L + 1):
            run, cur = [], s0
            for _ in range(len(col)):
                run.append(cur)
                cur = cur % L + 1
            if sorted(run) == sorted(s for _, _, s in col):
                starts.append(s0)
        lead = want if want in starts else (starts[0] if starts else want)
        keyed = sorted(((s - lead) % L, r, t, s) for r, t, s in col)
        out.extend((t, s) for _, _, t, s in keyed)
        prev = keyed[-1][3]
    return out


def numeral_price(dist):
    """What the recogniser charges to call this cell a numeral.

    Negative means it would rather. A cell with no centre crop has no
    opinion and costs nothing either way -- 23% of the ordinals produce
    no crop, so treating absence as evidence of absence would be a
    stronger claim than the corpus supports.
    """
    if not dist:
        return 0.0
    return (min(dist.get(k, BIG) for k in NUMERALS) -
            min(dist.get(k, BIG) for k in NONE_LIKE))


def inning_playable(gt, facts, seq, feasible):
    """The occupancy test for one candidate inning.

    Same shape as selfcheck's inner loop: the marked arrivals have to
    admit at least one legal set of runner agendas. A segment whose
    agenda product exceeds the cap is called playable rather than
    pruned -- never delete what has not been judged.
    """
    seq_facts = [facts[tag] for tag, _ in seq]
    if USE_MARKS < 1:
        return feasible(seq_facts, {})
    slot_pos = {}
    for k, (_, slot) in enumerate(seq):
        slot_pos.setdefault(str(slot), []).append(k)
    per_runner = []
    for k, (tag, _) in enumerate(seq):
        adv = gt["cells"].get(tag, {}).get("advances") or {}
        marks = []
        for bname, bnum in (("2B", 2), ("3B", 3), ("home", 4)):
            if bname not in adv:
                continue
            v = adv[bname]
            if USE_MARKS < 2 or v in ("PB", "S"):
                marks.append([(bnum, None)])   # arrival play unknown
            else:
                marks.append([(bnum, q) for q in slot_pos.get(str(v), [])
                              if q > k])
        if marks:
            per_runner.append((k, marks))
    if not per_runner:
        return feasible(seq_facts, {})
    tried = 0
    for combo in itertools.product(
            *(itertools.product(*m) for _, m in per_runner)):
        tried += 1
        if tried > COMBO_CAP:
            return True
        agendas = {p: tuple(c) for (p, _), c in zip(per_runner, combo)}
        if all(all(a[i][1] is None or a[i + 1][1] is None or
                   a[i][1] < a[i + 1][1] for i in range(len(a) - 1))
               for a in agendas.values()) and feasible(seq_facts, agendas):
            return True
    return False


# The header line score, as a price on the LENGTH of an inning.
#   A closed inning is  batters = 3 outs + runs + left on base,  so with
#   R read off the header, m - 3 - R is the number of runners stranded
#   and has to sit in [0, 3]. Nothing in that reads a corner or a reach;
#   it is arithmetic on how many cells the cut puts in the inning. The
#   last segment may be cut off mid-inning, so it is allowed [-3, 3].
#   Measured on the ledger before it was used: 97 of 98 closed innings
#   on the 17 sheets that carry a line score sit inside the bound, and
#   the one that does not is the inning whose third out the scorer never
#   wrote down. So it is a price, not a gate.
RUNS = os.environ.get("CORNER_SPLIT_RUNS", "0") == "1"


# The last hard gate in the model, and the one fact it stands on.
#
# A segment was admissible only if it held at most three batter outs.
# That is a GATE on `facts[tag]["base"] is None`, i.e. on the reach
# fact -- and with reach STAGED it is free, but the reach recogniser is
# 0.847 accurate, so one flipped cell can delete every correct cut of
# an inning. Measured on this corpus, ledger-free: with reach staged
# the split gets 140 of 175 innings and the corpus scores 328; with
# reach predicted and nothing else changed, 55 and 281.
#
# So do to it what was already done to the ordinal gate: make it a
# price. CORNER_SPLIT_OUTS raises the cap; segments over three pay
# `batter outs - 3` in a term of their own, compared FIRST, so a
# partition made only of three-out innings still beats any partition
# that needs a fourth. Default 3 = the gate, unchanged.
OUT_CAP = int(os.environ.get("CORNER_SPLIT_OUTS", "3"))


def runs_price(runs, m, k, last):
    """What a segment of m cells pays for being the k-th inning."""
    if not runs:
        return 0.0
    R = runs.get(str(k))
    if R is None:
        return 0.0
    lob = m - 3 - R
    lo, hi = (-3, 3) if last else (0, 3)
    if lo <= lob <= hi:
        return 0.0
    return float(lo - lob if lob < lo else lob - hi)


_DEF = ("o" if OUT_CAP > 3 else "") + ("r" if RUNS else "") + "tmp"
ORDER = os.environ.get("CORNER_SPLIT_ORDER", _DEF)
INF = (BIG,) * len(ORDER)

# tup() only emits the letters ORDER names, so a cap raised
# without an "o" to charge it is a gate quietly loosened for
# free. Same guard 65f228a gave "r".
if OUT_CAP > 3 and "o" not in ORDER:
    raise ValueError("CORNER_SPLIT_OUTS>3 needs an o in "
                     "CORNER_SPLIT_ORDER")
# How the named-batter term is charged.
#   1  one-sided: a corner costs 1 when the slot it names is not among
#      the later cells of its segment.
#   2  two-sided: the mark names ONE batter, so a segment should offer
#      it exactly one candidate; charge abs(candidates - 1). The second
#      side is what a segment pays for running long enough to lap the
#      order and hand the mark a second cell with the same slot.
# Declared before it was measured. The audit that asked for this said
# the one-sided term can only fall as a segment grows; named_monotone
# below measures that on the corpus instead of assuming it.
TWO_SIDED = os.environ.get("CORNER_SPLIT_NAMED", "1") == "2"


def mark_table(flat, named):
    """[(cell index, positions of the slot it names, after it)]."""
    pos_of = collections.defaultdict(list)
    for k, (_, slot) in enumerate(flat):
        pos_of[str(slot)].append(k)
    out = []
    for k, (tag, _) in enumerate(flat):
        for want in (named or {}).get(tag, ()):
            out.append((k, [q for q in pos_of.get(want, ()) if q > k]))
    return out


def named_cost(marks, i, j):
    """What the corners inside [i, j) pay for the segment they land in."""
    tot = 0
    for k, ps in marks:
        if i <= k < j:
            c = bisect.bisect_left(ps, j)
            tot += abs(c - 1) if TWO_SIDED else (1 if c == 0 else 0)
    return tot


def named_monotone(flat, named):
    """Which way does the ONE-SIDED count move as a segment is grown?

    Growing [i, j) to [i, j + 1) hands the new cell its own marks with
    an empty window after them, so the count can rise before any
    earlier cell is resolved. Walk every (i, j) on the sheet and count
    the steps each way, so the premise behind the two-sided term is
    measured rather than assumed.

    Counted over EVERY (i, j) on the sheet, including segments the
    DP never prices (more batter outs than the cap, nothing
    playable), so these are counts over the term, not over
    the chosen cut.
    """
    marks = mark_table(flat, named)
    up = down = same = 0
    for i in range(len(flat)):
        prev = None
        for j in range(i + 1, len(flat) + 1):
            tot = sum(1 for k, ps in marks
                      if i <= k < j and bisect.bisect_left(ps, j) == 0)
            if prev is not None:
                up += tot > prev
                down += tot < prev
                same += tot == prev
            prev = tot
    return up, down, same


# A price term that is always zero would look exactly like a term that
# works, so the split reports what each of its three terms actually paid.
STATS = {"segments": 0, "trail": 0, "named": 0, "violations": 0,
         "priced": 0, "mono_up": 0, "mono_down": 0, "mono_same": 0, "overout": 0}


def split(gt, facts, flat, prices, feasible, named=None,
          runs=None):
    """flat -> {inning: [(tag, slot)]}, or None if nothing is playable.

    facts: the same {tag: {"base": ...}} the solver uses, so this reads
    predicted reach when the reach predictor is on and staged reach
    otherwise. prices: {tag: numeral price}. named: {tag: [slot, ...]}
    the batting slots this cell's corners are RECOGNISED to name, which
    have to be in the same inning and later in it -- the constraint that
    the GT-marks diagnostic showed is worth 15 points end to end. It is
    counted as violations, not gated: the raw corner read is right about
    76% of the time, and a gate on a 76% signal deletes true segments.

    runs: the header line score, {inning: runs}, used only when
    CORNER_SPLIT_RUNS=1 -- a transcribed fact, so it is off by
    default. It prices an inning's LENGTH (see runs_price) and
    makes the DP carry how many segments are already closed,
    because the line score numbers the innings.

    CORNER_SPLIT_ORDER names the price terms in the order they are
    compared: t = cells trailing the last batter out, m = named-batter
    violations, p = the ordinal recogniser's charge, r = the
    line-score length price when CORNER_SPLIT_RUNS is on, and o =
    batter outs past three when CORNER_SPLIT_OUTS raises the cap.
    The default order gains each letter as its term is switched on.
    The default "tmp"
    is chosen by how reliable each rule is on its own -- 140 innings of
    143 end at their last out, the raw corner label is right on 76% of
    corners, the centre numeral on 62% of ordinal cells -- and the
    alternatives are measured in cv/README.md rather than picked from.
    """
    n = len(flat)
    if not n:
        return None
    is_out = [facts.get(t, {}).get("base") is None for t, _ in flat]
    pre = [0]
    for o in is_out:
        pre.append(pre[-1] + (1 if o else 0))
    pick = [None if is_out[k] else prices.get(t, 0.0)
            for k, (t, _) in enumerate(flat)]
    marks = mark_table(flat, named)
    _u, _d, _e = named_monotone(flat, named)
    STATS["mono_up"] += _u
    STATS["mono_down"] += _d
    STATS["mono_same"] += _e
    cost, seg_terms = {}, {}
    for i in range(n):
        for j in range(i + 1, n + 1):
            b = pre[j] - pre[i]
            if b > OUT_CAP:
                continue
            opts = sorted(p for p in pick[i:j] if p is not None)
            need = max(0, 3 - b)
            if j == n:                       # last inning may be cut off
                price = sum(p for p in opts[:need] if p < 0)
            elif len(opts) >= need:
                price = sum(opts[:need])
            else:
                continue
            if not inning_playable(gt, facts, flat[i:j], feasible):
                continue
            idx = [k for k in range(i, j) if is_out[k]]
            trail = float(j - 1 - idx[-1]) if idx else float(j - i)
            bad = float(named_cost(marks, i, j))
            cost[(i, j)] = {"t": trail, "m": bad, "p": price,
                            "o": float(max(0, b - 3))}
            seg_terms[(i, j)] = (trail, bad, price)
    def tup(part, r=0.0):
        d = dict(part)
        d["r"] = r
        d.setdefault("o", 0.0)
        return tuple(d[c] for c in ORDER)

    if RUNS and runs:
        # The line score numbers the innings, so the cut has to know
        # which one it is building: the state carries how many segments
        # are already closed.
        # A sheet needing more segments than this returns None and
        # the caller keeps the ledger -- a fallback the 1-D
        # branch does not have. Never taken on this corpus
        # (every log reports "0 sheets fell back").
        kmax = len(runs) + 3
        best = [[INF] * (kmax + 1) for _ in range(n + 1)]
        back = [[None] * (kmax + 1) for _ in range(n + 1)]
        best[0][0] = (0.0,) * len(ORDER)
        for j in range(1, n + 1):
            for k in range(1, kmax + 1):
                for i in range(j):
                    part = cost.get((i, j))
                    if part is None or best[i][k - 1] >= INF:
                        continue
                    c = tup(part, runs_price(runs, j - i, k, j == n))
                    v = tuple(a + b for a, b in zip(best[i][k - 1], c))
                    if v < best[j][k]:
                        best[j][k], back[j][k] = v, i
        if "r" not in ORDER:
            raise ValueError("CORNER_SPLIT_RUNS=1 needs an "
                             "r in CORNER_SPLIT_ORDER")
        ri = ORDER.index("r")
        bk, bv = None, INF
        for k in range(1, kmax + 1):
            v = best[n][k]
            if v >= INF:
                continue
            # a sheet that batted len(runs) innings has to be cut into
            # that many pieces; the shortfall is charged in the same
            # units as the length price
            w = list(v)
            w[ri] += abs(k - len(runs))
            w = tuple(w)
            if w < bv:
                bv, bk = w, k
        if bk is None:
            return None
        cuts, j, k = [], n, bk
        while j:
            i = back[j][k]
            cuts.append((i, j))
            j, k = i, k - 1
    else:
        best = [INF] * (n + 1)
        back = [None] * (n + 1)
        best[0] = (0.0,) * len(ORDER)
        for j in range(1, n + 1):
            for i in range(j):
                part = cost.get((i, j))
                if part is None or best[i] >= INF:
                    continue
                c = tup(part)
                v = tuple(a + b for a, b in zip(best[i], c))
                if v < best[j]:
                    best[j], back[j] = v, i
        if best[n] >= INF:
            return None
        cuts, j = [], n
        while j:
            cuts.append((back[j], j))
            j = back[j]
    seqs = {}
    for k, (i, j) in enumerate(reversed(cuts)):
        seqs[k + 1] = list(flat[i:j])
        t_, b_, p_ = seg_terms[(i, j)]
        STATS["segments"] += 1
        STATS["trail"] += t_ > 0
        STATS["named"] += b_ > 0
        STATS["violations"] += b_
        STATS["priced"] += abs(p_) > 1e-12
        STATS["overout"] += cost[(i, j)]["o"] > 0
    return seqs


def prices_leave_one_sheet_out(samples):
    """{(gid, tag): numeral price} under out_detect's own fold."""
    import corner_knn
    import out_detect
    if not samples:
        return {}
    feats = corner_knn.corner_feats([v for _, _, v, _ in samples])
    gids = [g for g, _, _, _ in samples]
    labs = [l for _, _, _, l in samples]
    out = {}
    for i, (gid, tag, _, _) in enumerate(samples):
        pool = collections.defaultdict(list)
        for j in range(len(samples)):
            if gids[j] != gid:
                pool[labs[j]].append(corner_knn.corner_dist(feats, i, j))
        d = {lab: sum(sorted(ds)[:out_detect.MNN]) /
             len(sorted(ds)[:out_detect.MNN]) for lab, ds in pool.items()}
        out[(gid, tag)] = numeral_price(d)
    return out
