"""A small web page: photo of a scorebook sheet -> game result and batting lines, with what to check.

Usage (repo root):
  python -m scorebook.web [--port 8765]

- Stored sheets (eval/manifests/games33.json) replay saved Claude replies: no key, no cost.
- Uploading a new photo calls the Anthropic API (SCOREBOOK_API_KEY or ANTHROPIC_API_KEY):
  13 calls for the columns + 1 for the line-score box, about 20 s; "read twice" doubles the column calls.
  Replies are cached under runs/api-cache/ (keyed by the photo's md5), so a reload costs nothing.

What the page shows, and why (experiments/*/RESULTS.md, measured on cloud):
- the line score computed from the cells AND the line score the scorer wrote in the box at the
  top, side by side, with the innings that differ marked. Neither is picked automatically: when
  they differ the computed one was wrong on 8/8 sheets, but the box also differs on 7/25 sheets
  whose computed line score is right (tiebreaks, a partly filled box, a slip of the pen).
- cells where two independent reads disagree (only when a second read exists): 61% of those
  cells were wrong against 6% of the others; no disagreement does NOT mean correct.
- the grid image without the name column (names are never shown).
"""
import argparse
import base64
import concurrent.futures as cf
import hashlib
import html
import json
import os
import sys
import urllib.parse
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from . import game, grid, read
from .__main__ import REPLAY_CORNERS, REPLAY_FACTS_V2, ROOT

# lineup replies hold real names: private repo only (the public export has no replies)
LINEUP_DIR = os.path.join(ROOT, "experiments/lineup_2026-10-22/runs/lineup")
# stored sheets: (label, manifest, facts_v2 replies, second read or None, header replies)
STORED = [
    ("開発用の 33 枚（2 回分の読み）", os.path.join(ROOT, "eval/manifests/games33.json"), REPLAY_FACTS_V2,
     os.path.join(ROOT, "experiments/facts_v2_repeat_2026-10-21/runs/facts_v2"),
     os.path.join(ROOT, "experiments/header_2026-10-21/runs/header"), REPLAY_CORNERS, LINEUP_DIR),
    ("評価に使っていない 2 枚（2026-10-18）", os.path.join(ROOT, "eval/manifests/heldout_2026-10-18.json"),
     os.path.join(ROOT, "experiments/heldout_v2_2026-10-21/runs/facts_v2"), None,
     os.path.join(ROOT, "experiments/heldout_v2_2026-10-21/runs/header"),
     os.path.join(ROOT, "experiments/heldout_2026-10-18/runs/raw1"), LINEUP_DIR),
]
CACHE = os.path.join(ROOT, "runs", "api-cache")
UPLOADS = os.path.join(ROOT, "runs", "uploads")
WORKERS = 6
# for use beyond this machine (e.g. the team's phones through a tunnel):
#   SCOREBOOK_WEB_PASSWORD - every page asks for it (HTTP Basic; any user name)
#   SCOREBOOK_DAILY_SHEETS - at most this many new photos read with the API per day (default 20)
PASSWORD = os.environ.get("SCOREBOOK_WEB_PASSWORD", "")
DAILY_SHEETS = int(os.environ.get("SCOREBOOK_DAILY_SHEETS", "20"))
USAGE = os.path.join(ROOT, "runs", "web-usage.json")


def sheets_today(add=0):
    """New photos read with the API today (cached photos do not count)."""
    import datetime
    today = datetime.date.today().isoformat()
    try:
        u = json.load(open(USAGE))
    except (OSError, ValueError):
        u = {}
    n = u.get(today, 0) + add
    if add:
        os.makedirs(os.path.dirname(USAGE), exist_ok=True)
        json.dump({today: n}, open(USAGE, "w"))
    return n


# ---------------------------------------------------------------- reading

def stored_sheets():
    """{sheet: (group label, photo, facts dir, second-read dir, header dir)} for sheets whose
    replies are stored (private repo only; the public copy has none)."""
    out = {}
    for label, manifest, fdir, f2dir, hdir, cdir, ldir in STORED:
        if not (os.path.exists(manifest) and os.path.isdir(fdir)):
            continue
        for g in json.load(open(manifest)):
            sheet = os.path.basename(g["gt"])[:-5]
            out[sheet] = (label, os.path.join(ROOT, g["image"]), fdir,
                          f2dir if f2dir and os.path.isdir(f2dir) else None,
                          hdir if os.path.isdir(hdir) else None,
                          cdir if os.path.isdir(cdir) else None,
                          ldir if os.path.isdir(ldir) else None)
    return out


def read_columns(reader, kind, sheet, cols):
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        cells = list(ex.map(lambda c: reader.read(kind, sheet, c), cols))
    return [(c.index, cl, {}) for c, cl in zip(cols, cells)]


def disagreements(r1, r2):
    """Cells where two reads differ: [(column, slot, what)]."""
    out = []
    for (col, a, _), (_, b, _) in zip(r1, r2):
        a, b = a or {}, b or {}
        for key in sorted(set(a) | set(b), key=lambda k: int(str(k).lstrip("r") or 0)):
            ca, cb = a.get(key) or {}, b.get(key) or {}
            if not isinstance(ca, dict) or not isinstance(cb, dict):
                continue
            if not ca.get("pa") and not cb.get("pa"):
                continue
            if bool(ca.get("pa")) != bool(cb.get("pa")):
                diff = ["pa"]
            else:
                diff = (["result"] if ca.get("result") != cb.get("result") else []) + [
                    f for f in ("scored", "strikeout") if bool(ca.get(f)) != bool(cb.get(f))]
            if diff:
                out.append((col, int(str(key).lstrip("r")), diff))
    return out


def box_rows(rows):
    out = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        inn = []
        for v in r.get("innings") or []:
            if isinstance(v, int) and not isinstance(v, bool):
                inn.append(v)
            elif v is None:
                inn.append(None)
            else:
                s = str(v).strip()
                inn.append(int(s) if s.isdigit() else s)
        if inn:
            out.append({"innings": inn, "total": r.get("total")})
    return out


def grid_image(photo):
    """The scoring grid only (inning columns x batter rows): the name column is left out."""
    img, cols, rows = grid.detect(photo)
    m = grid.MARGIN
    crop = img[max(0, rows[0][0] - m):rows[-1][1] + m, max(0, cols[0][0] - m):cols[-1][1] + m]
    scale = 1400 / max(crop.shape[:2])
    if scale < 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode()


def lineup_hints(slots):
    """{slot: (hint text, substitute note)} from a lineup reply. Names are hints only: matched to
    the site's player on 73% of rows (experiments/lineup_2026-10-22); the player count per slot on 98%."""
    out = {}
    for s in slots or []:
        if not isinstance(s, dict) or not isinstance(s.get("slot"), int):
            continue
        ps = [p for p in s.get("players") or [] if isinstance(p, dict)]
        names = [str(p.get("name") or (f"#{p['jersey']}" if p.get("jersey") is not None else "?")) for p in ps]
        note = ""
        if len(ps) > 1:
            when = ["%s回から" % p["from_inning"] if isinstance(p.get("from_inning"), int) else "回は不明" for p in ps[1:]]
            note = "交代あり（" + "、".join(f"{k + 2} 人目は{w}" for k, w in enumerate(when)) + "）"
        out[s["slot"]] = (" → ".join(names), note)
    return out


def analyse(photo, sheet, reader, reader2=None, header_reader=None):
    cols = grid.columns(photo)
    with cf.ThreadPoolExecutor(3) as ex:  # the reads run at the same time
        f1 = ex.submit(read_columns, reader, "facts_v2", sheet, cols)
        fc = ex.submit(read_columns, reader, "corners", sheet, cols)  # for RBI and stolen bases
        f2 = ex.submit(read_columns, reader2, "facts_v2", sheet, cols) if reader2 else None
        r1, rc, r2 = f1.result(), fc.result(), (f2.result() if f2 else None)
    # "scored" comes from facts_v2 (--scored read); the corners only feed RBI and stolen bases
    pas = game.assemble([(c, facts, corners or {}) for (c, facts, _), (_, corners, _) in zip(r1, rc)], "read")
    res_has_corners = any(corners is not None for _, corners, _ in rc)
    res = game.summary(pas)
    res["sheet"] = sheet
    res["columns"] = len(cols)
    res["unread_columns"] = [c for c, cells, _ in r1 if cells is None]
    res["disagreements"] = disagreements(r1, r2) if r2 is not None else None
    res["corners_read"] = res_has_corners
    res["box"] = None
    if header_reader:
        rows = header_reader.read("header", sheet, grid.Column("h", grid.header(photo), 0))
        res["box"] = box_rows(rows) if rows is not None else None
    res["lineup"] = {}
    dirs = getattr(reader, "dirs", None)  # ReplayReader: read only when lineup replies are stored
    if dirs is None or dirs.get("lineup"):
        try:
            res["lineup"] = lineup_hints(reader.read("lineup", sheet, grid.Column("l", grid.lineup(photo), 0)))
        except (Exception, SystemExit):
            res["lineup"] = {}
    res["image"] = grid_image(photo)
    return res


# ---------------------------------------------------------------- page

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6a66;--line:#d9d6cf;--acc:#2f6f4f;--warn:#b0451f;--warnbg:#fbeee8}
@media (prefers-color-scheme:dark){:root{--bg:#171716;--fg:#ecebe7;--mut:#a3a19b;--line:#3a3936;--acc:#7cc3a0;--warn:#f0936e;--warnbg:#3a2620}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px 16px 60px}h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px}
.mut{color:var(--mut)}a{color:var(--acc)}table{border-collapse:collapse;font-variant-numeric:tabular-nums}
td,th{border:1px solid var(--line);padding:3px 8px;text-align:center}th{font-weight:600}.scroll{overflow-x:auto}
td.diff{background:var(--warnbg);color:var(--warn);font-weight:700}.warn{color:var(--warn)}
.box{border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:10px 0}
img{max-width:100%;border:1px solid var(--line);border-radius:6px}ul{padding-left:20px}
form{display:flex;gap:10px;flex-wrap:wrap;align-items:center}button{font:inherit;padding:6px 14px}
input.nm{font:inherit;width:7em;background:transparent;color:inherit;border:1px solid var(--line);border-radius:4px;padding:1px 4px}
hr{border:0;border-top:2px solid var(--line);margin:36px 0}
div.sub{font-size:12px;color:var(--warn);white-space:nowrap}input.nm::placeholder{color:var(--mut);opacity:.8}
"""


def page(title, body):
    return ("<!doctype html><html lang=ja><head><meta charset=utf-8>"
            "<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
            f"<body><main>{body}</main></body></html>").encode()


def fmt(x):
    return "-" if x is None else ("%.3f" % x).lstrip("0") if x < 1 else "%.3f" % x


def line_score_html(res):
    comp = res["line_score"]
    box = res["box"] or []
    n = max([len(comp)] + [len(r["innings"]) for r in box])
    head = "".join(f"<th>{i + 1}</th>" for i in range(n))
    # Which box row is this sheet's team is not read (the box lists the visitor on top). Differences
    # are marked against the row whose total is closest to the computed total (then the closest
    # innings); the page says so, since it can pick the wrong row.
    own = None
    if box:
        def ints(v):
            return [x if isinstance(x, int) else 0 for x in v]

        def key(r):
            inn = ints(r["innings"])
            tot = r["total"] if isinstance(r.get("total"), int) else sum(inn)
            l1 = sum(abs((inn[i] if i < len(inn) else 0) - (comp[i] if i < len(comp) else 0)) for i in range(n))
            return (abs(tot - sum(comp)), l1)
        own = min(box, key=key)

    def row(label, vals, other=None, total=None):
        cells = []
        for i in range(n):
            v = vals[i] if i < len(vals) else ""
            o = (other[i] if i < len(other) else "") if other is not None else None
            mark = other is not None and str(v) != str(o) and not (v == "" and o in ("", None))
            cells.append(f"<td class={'diff' if mark else ''}>{'' if v is None else html.escape(str(v))}</td>")
        tot = sum(v for v in vals if isinstance(v, int)) if total is None else total
        return f"<tr><th style='text-align:left'>{label}</th>{''.join(cells)}<td>{'' if tot is None else tot}</td></tr>"

    rows = [row("マスから計算", comp, own["innings"] if own else None)]
    src = f"（同じ試合のもう 1 枚 {html.escape(res['box_from'])} の用紙から）" if res.get("box_from") else ""
    for k, r in enumerate(box):
        lab = "用紙の得点表・" + ("上の行（先攻）" if k == 0 else "下の行（後攻）") + ("＝比べた行" if r is own else "")
        rows.append(row(lab, r["innings"], comp if r is own else None, r.get("total")))
    note = ""
    if res["box"] is None:
        note = "<p class=mut>用紙の得点表は読んでいません。</p>"
    elif not box:
        note = "<p class=mut>この用紙の上の得点表は空です（同じ試合のもう 1 枚に書かれていることが多い）。</p>"
    elif own and [v for v in own["innings"]] != comp:
        note = ("<p class=warn>計算と用紙の得点表（合計が近い方の行と比べています）が違う回があります（色つき）。"
                "測った 33 枚では、違うときは計算が外れていることが多い一方、タイブレークや書きかけで用紙側が違うこともあります。"
                "用紙を見て確かめてください。</p>")
    else:
        note = "<p class=mut>計算と用紙の得点表は一致しています。</p>"
    return (f"<div class=scroll><table><tr><th></th>{head}<th>計</th></tr>{''.join(rows)}</table></div>"
            + (f"<p class=mut>用紙の得点表{src}</p>" if src else "") + note)


BAT_HEAD = ("打順", "名前", "打席", "打数", "安打", "二塁打", "三塁打", "本塁打", "打点", "四球", "死球", "失策出塁", "三振",
            "盗塁", "得点", "打率", "出塁率", "長打率")


def batting_html(res, k=0):
    """Batting lines. The name column is typed by the user and stays in the browser (never sent)."""
    head = "".join(f"<th>{h}</th>" for h in BAT_HEAD)
    rows = []
    for b in res["batting"]:
        hint, note = res.get("lineup", {}).get(b["slot"], ("", ""))
        rows.append(f"<tr><td>{b['slot']}</td><td><input class=nm data-slot={b['slot']} "
                    f"placeholder='{html.escape(hint, quote=True)}' "
                    f"aria-label='打順 {b['slot']} の名前'>"
                    + (f"<div class=sub>{html.escape(note)}</div>" if note else "") + "</td>" + "".join(f"<td>{v}</td>" for v in (
                        b["PA"], b["AB"], b["H"], b["doubles"], b["triples"], b["HR"], b["RBI"], b["BB"], b["HBP"],
                        b["ROE"], b["SO"], b["SB"], b["R"], fmt(b["AVG"]), fmt(b["OBP"]), fmt(b["SLG"]))) + "</tr>")
    name = html.escape(res["sheet"])
    return (f"<div class=scroll><table id=bat{k} data-sheet='{name}'><tr>{head}</tr>{''.join(rows)}</table></div>"
            f"<p><button type=button onclick=\"csv('bat{k}')\">CSV で保存</button> "
            "<span class=mut>名前はこのブラウザの中だけで使い、サーバーにも Claude にも送りません。</span></p>")


def checks_html(res):
    items = []
    if res["unread_columns"]:
        items.append("読めなかった列：" + "、".join(f"{c + 1} 列目" for c in res["unread_columns"]))
    d = res["disagreements"]
    if d is None:
        items.append("<span class=mut>2 回目の読みはありません（「2 回読む」で、読みが割れたマスを示せます）。</span>")
    elif d:
        names = {"pa": "打席の有無", "result": "結果", "scored": "得点", "strikeout": "三振"}
        items.append("2 回の読みが割れたマス（ここは誤りが多い）：<ul>" + "".join(
            f"<li>{c + 1} 列目・{s} 番打者：{'・'.join(names[w] for w in what)}</li>" for c, s, what in d) + "</ul>")
    else:
        items.append("2 回の読みは全部のマスで一致しました（一致していても誤りはありえます）。")
    return "<div class=box>" + "".join(f"<p>{i}</p>" for i in items) + "</div>"


JS = '\nfunction cell(t){t=String(t);return /[",\\n]/.test(t)?\'"\'+t.replace(/"/g,\'""\')+\'"\':t}\nfunction csv(id){\n  var t=document.getElementById(id),out=[];\n  t.querySelectorAll(\'tr\').forEach(function(tr){\n    var r=[];tr.querySelectorAll(\'th,td\').forEach(function(c){var i=c.querySelector(\'input\');r.push(cell(i?i.value:c.textContent))});\n    out.push(r.join(\',\'))});\n  var b=new Blob([\'\ufeff\'+out.join(\'\\n\')],{type:\'text/csv\'}),a=document.createElement(\'a\');\n  a.href=URL.createObjectURL(b);a.download=(t.dataset.sheet||\'scorebook\')+\'.csv\';a.click()}\n// names typed for a batting slot are remembered in this browser only (a convenience)\ndocument.querySelectorAll(\'input.nm\').forEach(function(i){\n  var k=\'sb-name-\'+i.closest(\'table\').dataset.sheet+\'-\'+i.dataset.slot;\n  try{i.value=localStorage.getItem(k)||\'\'}catch(e){}\n  i.addEventListener(\'input\',function(){try{localStorage.setItem(k,i.value)}catch(e){}})});\n'


SITE_COLS = ("打順", "交代", "打席", "打数", "安打", "2B", "3B", "HR", "打点", "四死", "犠", "盗", "三振")


def own_box_row(res):
    """(index of this team's row in the box, the row) - chosen by the closest total, as on the page."""
    box, comp = res.get("box") or [], res["line_score"]
    if not box:
        return None, None

    def key(r):
        inn = [x if isinstance(x, int) else 0 for x in r["innings"]]
        tot = r["total"] if isinstance(r.get("total"), int) else sum(inn)
        return (abs(tot - sum(comp)), sum(abs((inn[i] if i < len(inn) else 0) - (comp[i] if i < len(comp) else 0))
                                         for i in range(max(len(inn), len(comp)))))
    i = min(range(len(box)), key=lambda j: key(box[j]))
    return i, box[i]


def site_html(res, partner=None):
    """What to type into the team site (yokohama-funnies: /edit/results and /edit/game-stats)."""
    i, own = own_box_row(res)
    us = ",".join(str(v) for v in res["line_score"])
    them, them_src = "", ""
    if own is not None and len(res["box"]) > 1:
        other = res["box"][1 - i] if i in (0, 1) else None
        if other:
            them = ",".join("" if v is None else str(v) for v in other["innings"])
            them_src = "用紙の得点表のもう 1 行"
    elif partner is not None:
        them = ",".join(str(v) for v in partner["line_score"])
        them_src = f"もう 1 枚（{partner['sheet']}）から計算"
    side = "" if own is None else ("後攻（ホーム）" if i == 1 else "先攻（ビジター）")
    box_us = ",".join("" if v is None else str(v) for v in own["innings"]) if own else ""
    lines = [f"<tr><th style='text-align:left'>この用紙のチーム（マスから計算）</th><td><code>{us}</code></td></tr>"]
    if box_us and box_us != us:
        lines.append(f"<tr><th style='text-align:left'>この用紙のチーム（用紙の得点表）</th><td class=diff><code>{html.escape(box_us)}</code></td></tr>")
    if them:
        lines.append(f"<tr><th style='text-align:left'>相手（{html.escape(them_src)}）</th><td><code>{html.escape(them)}</code></td></tr>")
    if side:
        lines.append(f"<tr><th style='text-align:left'>先攻／後攻（用紙の得点表の行から）</th><td>{side}</td></tr>")
    rows = []
    for b in res["batting"]:
        rbi, sb = (b["RBI"], b["SB"]) if res.get("corners_read") else ("", "")
        vals = (b["slot"], 0, b["PA"], b["AB"], b["H"], b["doubles"], b["triples"], b["HR"], rbi, b["BB"] + b["HBP"], "", sb, b["SO"])
        rows.append("<tr>" + "".join(f"<td>{v}</td>" for v in vals) + "</tr>")
    return ("<p class=mut>試合結果の「イニング別スコア」と、試合別成績の入力欄の並びです。</p>"
            f"<div class=scroll><table>{''.join(lines)}</table></div>"
            f"<div class=scroll><table><tr>{''.join(f'<th>{h}</th>' for h in SITE_COLS)}</tr>{''.join(rows)}</table></div>"
            "<p class=mut>打点・盗塁は角の印から出しています（18 試合でサイトと一致したのは打順の行の 92%・93%）。"
            "犠打犠飛は読んでいないので空欄です。犠打犠飛は打数に入っているので、ある打者は打数を 1 減らしてください。"
            "途中交代は同じ打順にまとめています（交代は 0 のまま）。</p>")


def sheet_html(res, k, partner=None):
    return (f"<h1>{html.escape(res['sheet'])}</h1>"
            f"<p class=mut>{res['columns']} 列</p>"
            "<h2>試合結果（回ごとの得点）</h2>" + line_score_html(res) +
            "<h2>確認してほしいところ</h2>" + checks_html(res) +
            "<h2>打順ごとの成績</h2>" + batting_html(res, k) +
            "<h2>サイト入力用（横浜ファニーズ）</h2>" + site_html(res, partner) +
            "<p class=mut>名前の欄の薄い文字は、用紙から読んだ名前です（サイトの選手と合うのは 7 割ほどなので、確かめて入れてください）。"
            "途中交代は同じ打順にまとめています（交代がある打順には印が付きます）。</p>"
            "<h2>読んだ格子（名前の欄は写していません）</h2>"
            f"<img alt='scoring grid' src='data:image/jpeg;base64,{res['image']}'>")


def share_box(results):
    """The line-score box is often filled on only one sheet of the game: lend it to the other."""
    for r in results:
        if r.get("box") == []:
            for o in results:
                if o is not r and o.get("box"):
                    r["box"], r["box_from"] = o["box"], o["sheet"]
                    break


def result_page(results, source):
    share_box(results)
    body = (f"<p><a href='/'>← 戻る</a></p><p class=mut>{html.escape(source)}</p>" +
            "<hr>".join(sheet_html(r, k, results[1 - k] if len(results) == 2 else None)
                        for k, r in enumerate(results)) + f"<script>{JS}</script>")
    return page(" / ".join(r["sheet"] for r in results), body)


def index_page(has_key, msg=""):
    stored = stored_sheets()
    groups = {}
    for s in sorted(stored):
        groups.setdefault(stored[s][0], []).append(s)
    links = "".join(
        f"<li>{html.escape(label)}<ul>" + "".join(
            f"<li><a href='/sheet/{urllib.parse.quote(s)}'>{html.escape(s)}</a></li>" for s in ss) + "</ul></li>"
        for label, ss in groups.items()) or "<li class=mut>保存済みの用紙はありません。</li>"
    if has_key:
        upload = ("<form method=post action=/read enctype=multipart/form-data "
                  "onsubmit=\"this.querySelector('button').disabled=true;"
                  "document.getElementById('busy').hidden=false\">"
                  "<input type=file name=photo accept='image/*' multiple required>"
                  "<label><input type=checkbox name=twice> 2 回読む（確認用・費用 2 倍）</label>"
                  "<button>読む</button></form>"
                  "<p id=busy class=warn hidden>読んでいます（20 秒ほど）…</p>"
                  "<p class=mut>同じ試合の 2 枚（両チーム）をまとめて選べます。1 枚あたり 28 回（列 13 回×2＋得点表 1 回＋名前の欄 1 回、約 $0.27）の API 呼び出し、20 秒ほど。返答は保存し、同じ写真は二度と払いません。</p>")
    else:
        upload = "<p class=mut>API キー（SCOREBOOK_API_KEY）が無いので、新しい写真は読めません。下の保存済みの用紙は再生できます。</p>"
    body = ("<h1>スコアブックを撮るだけで、試合結果と成績</h1>"
            "<p class=mut>紙のスコアブック 1 枚の写真から、回ごとの得点と打順ごとの成績を出します。</p>"
            + (f"<p class=warn>{html.escape(msg)}</p>" if msg else "") +
            "<h2>新しい写真を読む</h2>" + upload +
            "<h2>保存済みの用紙（Claude の返答を再生・費用 0）</h2><ul>" + links + "</ul>")
    return page("スコアブック読み取り", body)


# ---------------------------------------------------------------- server

class Handler(BaseHTTPRequestHandler):
    def send(self, code, data, ctype="text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def allowed(self):
        if not PASSWORD:
            return True
        import hmac
        auth = self.headers.get("Authorization") or ""
        if auth.startswith("Basic "):
            try:
                given = base64.b64decode(auth[6:]).decode("utf-8", "replace").split(":", 1)[-1]
            except ValueError:
                given = ""
            if hmac.compare_digest(given.encode(), PASSWORD.encode()):
                return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="scorebook", charset="UTF-8"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def has_key(self):
        return bool(os.environ.get("SCOREBOOK_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))

    def do_GET(self):
        if not self.allowed():
            return
        path = urllib.parse.unquote(self.path.split("?")[0])
        if path == "/":
            return self.send(200, index_page(self.has_key()))
        if path.startswith("/sheet/"):
            sheet = path[len("/sheet/"):]
            found = stored_sheets().get(sheet)
            if not found:
                return self.send(404, page("not found", "<p>その用紙はありません。</p>"))
            label, photo, fdir, f2dir, hdir, cdir, ldir = found
            reader = read.ReplayReader(None, cdir, facts_v2=fdir, lineup=ldir)
            reader2 = read.ReplayReader(None, None, facts_v2=f2dir) if f2dir else None
            hreader = read.ReplayReader(None, None, header=hdir) if hdir else None
            try:
                res = analyse(photo, sheet, reader, reader2, hreader)
            except (Exception, SystemExit) as e:  # grid not found etc. (cv/ raises SystemExit)
                return self.send(500, page("error", f"<p class=warn>読めませんでした：{html.escape(str(e))}</p>"))
            if res["box"] == [] and hreader:  # the box is often filled on the other sheet of the game
                game_id = sheet.rsplit("_", 1)[0]
                for other in sorted(stored_sheets()):
                    if other != sheet and other.rsplit("_", 1)[0] == game_id:
                        rows = box_rows(hreader.read("header", other, grid.Column("h", b"", 0)))
                        if rows:
                            res["box"], res["box_from"] = rows, other
                            break
            return self.send(200, result_page([res], f"保存済みの返答を再生・{label}"))
        return self.send(404, page("not found", "<p>not found</p>"))

    def do_POST(self):
        if not self.allowed():
            return
        if self.path != "/read":
            return self.send(404, page("not found", "<p>not found</p>"))
        if not self.has_key():
            return self.send(200, index_page(False, "API キーが無いので読めません。"))
        n = int(self.headers.get("Content-Length") or 0)
        if n > 25 * 1024 * 1024:
            return self.send(200, index_page(True, "写真が大きすぎます（25 MB まで）。"))
        msg = BytesParser(policy=email_policy).parsebytes(
            b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + self.rfile.read(n))
        photos, twice = [], False
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if name == "photo":
                data = part.get_payload(decode=True)
                if data:
                    photos.append(data)
            elif name == "twice":
                twice = True
        if not photos:
            return self.send(200, index_page(True, "写真がありません。"))
        if len(photos) > 2:
            return self.send(200, index_page(True, "一度に読めるのは 2 枚（1 試合の両チーム）までです。"))
        jobs = []
        for k, data in enumerate(photos):
            sheet = "upload_" + hashlib.md5(data).hexdigest()[:12]
            os.makedirs(UPLOADS, exist_ok=True)
            photo = os.path.join(UPLOADS, sheet + ".jpg")
            with open(photo, "wb") as f:
                f.write(data)
            which = f"{k + 1} 枚目の" if len(photos) > 1 else ""
            try:
                cols = grid.columns(photo)
            except (Exception, SystemExit) as e:  # unreadable file or no grid (cv/ raises SystemExit)
                why = str(e).replace(photo, "写真")
                return self.send(200, index_page(True, f"{which}格子が見つかりませんでした（{why}）。用紙全体が入るように、正面から撮り直してください。"))
            if not 4 <= len(cols) <= 13:
                return self.send(200, index_page(True, f"{which}回の列が {len(cols)} 本しか見つかりませんでした。撮り直してください。"))
            jobs.append((photo, sheet))
        new = [sheet for _, sheet in jobs
               if not os.path.exists(os.path.join(CACHE, "facts_v2", sheet + "_c0.json"))]
        if new and sheets_today() + len(new) > DAILY_SHEETS:
            return self.send(200, index_page(True, f"今日読める枚数（{DAILY_SHEETS} 枚）を超えます。明日また試してください。"))
        sheets_today(add=len(new))
        reader = read.ApiReader(CACHE)
        reader2 = read.ApiReader(CACHE + "-2") if twice else None
        try:
            with cf.ThreadPoolExecutor(len(jobs)) as ex:
                results = list(ex.map(lambda j: analyse(j[0], j[1], reader, reader2, reader), jobs))
        except RuntimeError as e:
            return self.send(200, index_page(True, f"API の呼び出しに失敗しました：{e}"))
        except (Exception, SystemExit) as e:
            return self.send(200, index_page(True, f"読めませんでした（{e}）。"))
        return self.send(200, result_page(results, "API で読み取り" + ("（2 回）" if twice else "")))

    def log_message(self, fmt_, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt_ % args))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m scorebook.web")
    ap.add_argument("--port", type=int, default=8765)  # 8000 is taken on the owner's Pi
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args(argv)
    print(f"http://{a.host}:{a.port}/", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
