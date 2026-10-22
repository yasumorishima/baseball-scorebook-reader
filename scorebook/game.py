"""Assemble plate appearances from column readings, split innings, compute stats.

A sheet is one team's batting. Row rK of the grid is batting-order slot K.
"""
from dataclasses import asdict, dataclass, field

HITS = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
REACH = set(HITS) | {"walk", "hit_by_pitch", "error", "other_reach"}
RESULTS = REACH | {"out"}
NUMERALS = {"I": 1, "II": 2, "III": 3}


@dataclass
class PlateAppearance:
    slot: int
    column: int
    result: str  # one of RESULTS, or "unknown"
    out_numeral: int = 0  # 1..3 when the cell carries an out numeral (batter or runner)
    scored: bool = False
    strikeout: bool = False
    left_on_base: bool = False
    corners: dict = field(default_factory=dict)  # {"UR": .., "UL": .., "LL": ..}
    inning: int = 0
    suspect: str = ""  # why this PA's column looks wrong ("" = nothing found); shown for review only


def numeral_value(raw):
    if raw is None:
        return 0
    return NUMERALS.get(str(raw).strip().upper(), 0)


SCORED_MODES = ("auto", "corner", "read", "either")


def is_scored(cell, mark, mode="auto"):
    """corner: a mark at the home corner; read: the reader's own "scored" answer
    (facts_v2); either: one of the two; auto: read when the reply has it, else corner."""
    corner = mark.get("LL") not in (None, "")
    read = cell.get("scored")
    if mode == "auto":
        mode = "corner" if read is None else "read"
    if mode == "corner":
        return corner
    if mode == "read":
        return bool(read)
    return corner or bool(read)


def assemble(column_readings, scored_mode="auto", wrap_continues=False):
    """column_readings: list of (column_index, facts_cells, corners_cells) in column order.
    Returns the plate appearances with innings assigned."""
    pas = []
    for col, facts, corners in column_readings:
        facts = facts or {}
        corners = corners or {}
        col_pas = []
        for key, cell in facts.items():
            if not isinstance(cell, dict) or not cell.get("pa"):
                continue
            try:
                slot = int(str(key).lstrip("r"))
            except ValueError:
                continue
            result = cell.get("result")
            if result not in RESULTS:
                result = "unknown"
            mark = corners.get(key) or {}
            col_pas.append(PlateAppearance(
                slot=slot, column=col, result=result,
                out_numeral=numeral_value(cell.get("numeral")),
                scored=is_scored(cell, mark, scored_mode),
                strikeout=bool(cell.get("strikeout")),
                left_on_base=bool(cell.get("left_on_base")),
                corners={k: v for k, v in mark.items() if k in ("UR", "UL", "LL")},
            ))
        col_pas.sort(key=lambda p: p.slot)
        pas.append((col, col_pas))
    return split_innings(pas, wrap_continues)


def split_innings(columns, wrap_continues=False):
    """Give every plate appearance its inning.

    The scorer starts a new column for every inning, and carries an inning into the
    next column only when the order has batted around: the next batter's row in the
    current column is already used. So the next column continues the inning exactly
    when (a) fewer than three outs are marked so far and (b) the row of the batter due
    up is occupied in the previous column. Otherwise it is a new inning (this also
    covers innings closed without a third-out numeral, e.g. by a run limit).

    Within a column the batting order starts at the leadoff slot and wraps around,
    so a column can hold slots 7, 8, 9 and then 1, 2 of the same inning.

    wrap_continues (B2, experiments/assembly_2026-10-21, NOT adopted: it fixed one sheet and
    broke one whose inning ended at the last slot without a third-out numeral): the next column
    also continues the inning when the previous column ended with the last slot of the order.

    Columns that do not fit are marked on their plate appearances (p.suspect), never dropped
    (research only; not shown by the tool: the flag missed one ghost column and marked 11 others):
      "order"   - the batter due up is not in the column (the order does not connect)
      "numeral" - a new inning with out numerals but no I"""
    columns = [(c, ps) for c, ps in columns if ps]
    n = max((p.slot for _, ps in columns for p in ps), default=0)
    out = []
    inning, outs, due = 0, 0, 1
    prev_rows = set()
    for i, (col, col_pas) in enumerate(columns):
        wrapped = wrap_continues and i > 0 and due == 1
        cont = i > 0 and outs < 3 and (due in prev_rows or wrapped)
        if not cont:
            inning += 1
            outs = 0
        leadoff = due
        order = sorted(col_pas, key=lambda p: (p.slot - leadoff) % n)
        nums = {p.out_numeral for p in col_pas} - {0}
        suspect = ""
        if i > 0 and due not in {p.slot for p in col_pas}:
            suspect = "order"
        elif not cont and nums and 1 not in nums:
            suspect = "numeral"
        for p in order:
            p.inning = inning
            p.suspect = suspect
            out.append(p)
        outs = max([outs] + [p.out_numeral for p in col_pas])
        due = order[-1].slot % n + 1
        prev_rows = {p.slot for p in col_pas}
    return out


@dataclass
class BattingLine:
    slot: int
    PA: int = 0
    AB: int = 0
    H: int = 0
    doubles: int = 0
    triples: int = 0
    HR: int = 0
    BB: int = 0
    HBP: int = 0
    ROE: int = 0
    R: int = 0
    SO: int = 0
    RBI: int = 0  # from the corner marks (rbi_sb), 0 when corners were not read
    SB: int = 0
    unknown: int = 0

    def add(self, p):
        self.PA += 1
        r = p.result
        if r in ("walk", "hit_by_pitch"):
            self.BB += r == "walk"
            self.HBP += r == "hit_by_pitch"
        else:
            self.AB += 1
        if r in HITS:
            self.H += 1
            self.doubles += r == "double"
            self.triples += r == "triple"
            self.HR += r == "home_run"
        self.ROE += r == "error"
        self.unknown += r == "unknown"
        self.R += bool(p.scored)
        self.SO += bool(p.strikeout)

    def rates(self):
        avg = self.H / self.AB if self.AB else None
        obp = (self.H + self.BB + self.HBP) / self.PA if self.PA else None
        tb = self.H + self.doubles + 2 * self.triples + 3 * self.HR
        slg = tb / self.AB if self.AB else None
        return {"AVG": avg, "OBP": obp, "SLG": slg}


def rbi_sb(pas):
    """RBI and stolen bases per slot from the corner marks (experiments/rbi_sb_2026-10-22:
    92% / 93% of slots equal to the team site on 18 games).
    RBI of slot k = runners whose home (LL) corner says k in an inning where k batted, plus 1 for
    the batter's own home run (unless his LL already says k). SB = "S" marks at his corners."""
    rbi, sb = {}, {}
    batted = {}
    for p in pas:
        batted.setdefault(p.inning, set()).add(p.slot)
    for p in pas:
        c = p.corners or {}
        n = sum(1 for v in c.values() if v == "S")
        if n:
            sb[p.slot] = sb.get(p.slot, 0) + n
        home = c.get("LL")
        if isinstance(home, int) and home in batted.get(p.inning, ()):
            rbi[home] = rbi.get(home, 0) + 1
        if p.result == "home_run" and home != p.slot:
            rbi[p.slot] = rbi.get(p.slot, 0) + 1
    return rbi, sb


def batting(pas):
    lines = {}
    for p in pas:
        lines.setdefault(p.slot, BattingLine(p.slot)).add(p)
    rbi, sb = rbi_sb(pas)
    for slot, line in lines.items():
        line.RBI, line.SB = rbi.get(slot, 0), sb.get(slot, 0)
    return [lines[k] for k in sorted(lines)]


def line_score(pas):
    """Runs per inning for this sheet's team."""
    innings = max((p.inning for p in pas), default=0)
    runs = [0] * innings
    for p in pas:
        if p.scored:
            runs[p.inning - 1] += 1
    return runs


def suspect_columns(pas):
    """{column: reason} for columns flagged by split_innings."""
    return {p.column: p.suspect for p in pas if p.suspect}


def summary(pas):
    lines = batting(pas)
    runs = line_score(pas)
    return {
        "line_score": runs,
        "R": sum(runs),
        "H": sum(b.H for b in lines),
        "batting": [dict(asdict(b), **b.rates()) for b in lines],
        "plate_appearances": [asdict(p) for p in pas],
    }
