"""Read one scorebook photo into a game result and batting lines.

Usage:
  python -m scorebook <photo> --sheet ID [--replay] [--json out.json]
  python -m scorebook <photo> --sheet ID --api [--cache DIR]
  add --facts facts to use the older facts question (v1) instead of facts_v2

--replay serves stored Claude replies (no API key, no cost):
  facts_v2 from experiments/facts_v2_2026-10-21/runs/facts_v2/<ID>_c<col>.json (default)
  facts    from experiments/claude_facts_2026-10/runs/raw/<ID>_c<col>.json
  corners  from experiments/claude_vision_2026-10/raw/<ID>_c<col>.json
--api calls the Anthropic API (needs SCOREBOOK_API_KEY or ANTHROPIC_API_KEY); replies are cached under --cache.
facts_v2 also reads "scored" (centre circle), strikeout and left on base; "scored" then comes
from that answer instead of the home corner mark (facts_v2 PREREG, 2026-10-21).
"""
import argparse
import json
import os
import sys

from . import game, grid, read

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPLAY_FACTS_V1 = os.path.join(ROOT, "experiments/claude_facts_2026-10/runs/raw")
REPLAY_FACTS_V2 = os.path.join(ROOT, "experiments/facts_v2_2026-10-21/runs/facts_v2")
REPLAY_FACTS = REPLAY_FACTS_V2  # default facts replies
FACTS_KIND = "facts_v2"  # default facts question
REPLAY_CORNERS = os.path.join(ROOT, "experiments/claude_vision_2026-10/raw")


def run(photo, sheet, reader, facts_kind=FACTS_KIND):
    cols = grid.columns(photo)
    readings = []
    for c in cols:
        # facts_v2 answers "scored" itself, so the corners question is not asked (13 calls a sheet, not 26;
        # same numbers on 33 sheets, experiments/assembly_2026-10-21 C1)
        corners = reader.read("corners", sheet, c) if facts_kind == "facts" else {}
        readings.append((c.index, reader.read(facts_kind, sheet, c), corners))
    pas = game.assemble(readings)
    out = game.summary(pas)
    out["sheet"] = sheet
    out["columns"] = len(cols)
    return out


def fmt(x):
    return "-" if x is None else ("%.3f" % x).lstrip("0") if x < 1 else "%.3f" % x


def render(res):
    runs = res["line_score"]
    lines = ["sheet %s" % res["sheet"],
             "inning | " + " ".join("%2d" % (i + 1) for i in range(len(runs))) + " |  R  H",
             "runs   | " + " ".join("%2d" % r for r in runs) + " | %2d %2d" % (res["R"], res["H"]),
             "",
             "slot  PA AB  H 2B 3B HR BB HBP ROE SO  R   AVG   OBP   SLG"]
    for b in res["batting"]:
        lines.append("%4d %3d %2d %2d %2d %2d %2d %2d %3d %3d %2d %2d %5s %5s %5s" % (
            b["slot"], b["PA"], b["AB"], b["H"], b["doubles"], b["triples"], b["HR"],
            b["BB"], b["HBP"], b["ROE"], b["SO"], b["R"], fmt(b["AVG"]), fmt(b["OBP"]), fmt(b["SLG"])))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m scorebook")
    ap.add_argument("photo")
    ap.add_argument("--sheet", required=True, help="sheet id, used to name cached/stored replies")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--replay", action="store_true", help="use stored replies (default)")
    mode.add_argument("--api", action="store_true", help="call the Anthropic API")
    ap.add_argument("--facts", default=FACTS_KIND, choices=("facts_v2", "facts"),
                    help="which facts question to use (default facts_v2)")
    ap.add_argument("--cache", default=os.path.join(ROOT, "runs", "api-cache"))
    ap.add_argument("--json", help="write the full result as JSON here")
    a = ap.parse_args(argv)
    reader = read.ApiReader(a.cache) if a.api else read.ReplayReader(
        REPLAY_FACTS_V1, REPLAY_CORNERS, facts_v2=REPLAY_FACTS_V2)
    res = run(a.photo, a.sheet, reader, a.facts)
    print(render(res))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
