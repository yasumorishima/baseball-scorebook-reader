"""A small web page: photo of a scorebook sheet -> game result and batting lines, with what to check.

Usage (repo root):
  python -m scorebook.web [--port 8000]

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
from .__main__ import REPLAY_FACTS_V2, ROOT

# stored sheets: (label, manifest, facts_v2 replies, second read or None, header replies)
STORED = [
    ("開発用の 33 枚（2 回分の読み）", os.path.join(ROOT, "eval/manifests/games33.json"), REPLAY_FACTS_V2,
     os.path.join(ROOT, "experiments/facts_v2_repeat_2026-10-21/runs/facts_v2"),
     os.path.join(ROOT, "experiments/header_2026-10-21/runs/header")),
    ("評価に使っていない 2 枚（2026-10-18）", os.path.join(ROOT, "eval/manifests/heldout_2026-10-18.json"),
     os.path.join(ROOT, "experiments/heldout_v2_2026-10-21/runs/facts_v2"), None,
     os.path.join(ROOT, "experiments/heldout_v2_2026-10-21/runs/header")),
]
CACHE = os.path.join(ROOT, "runs", "api-cache")
UPLOADS = os.path.join(ROOT, "runs", "uploads")
WORKERS = 6


# ---------------------------------------------------------------- reading

def stored_sheets():
    """{sheet: (group label, photo, facts dir, second-read dir, header dir)} for sheets whose
    replies are stored (private repo only; the public copy has none)."""
    out = {}
    for label, manifest, fdir, f2dir, hdir in STORED:
        if not (os.path.exists(manifest) and os.path.isdir(fdir)):
            continue
        for g in json.load(open(manifest)):
            sheet = os.path.basename(g["gt"])[:-5]
            out[sheet] = (label, os.path.join(ROOT, g["image"]), fdir,
                          f2dir if f2dir and os.path.isdir(f2dir) else None,
                          hdir if os.path.isdir(hdir) else None)
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


def analyse(photo, sheet, reader, reader2=None, header_reader=None):
    cols = grid.columns(photo)
    with cf.ThreadPoolExecutor(2) as ex:  # the two reads run at the same time
        f1 = ex.submit(read_columns, reader, "facts_v2", sheet, cols)
        f2 = ex.submit(read_columns, reader2, "facts_v2", sheet, cols) if reader2 else None
        r1, r2 = f1.result(), (f2.result() if f2 else None)
    pas = game.assemble(r1)
    res = game.summary(pas)
    res["sheet"] = sheet
    res["columns"] = len(cols)
    res["unread_columns"] = [c for c, cells, _ in r1 if cells is None]
    res["disagreements"] = disagreements(r1, r2) if r2 is not None else None
    res["box"] = None
    if header_reader:
        rows = header_reader.read("header", sheet, grid.Column("h", grid.header(photo), 0))
        res["box"] = box_rows(rows) if rows is not None else None
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


def batting_html(res):
    head = "".join(f"<th>{h}</th>" for h in
                   ("打順", "打席", "打数", "安打", "二塁打", "三塁打", "本塁打", "四球", "死球", "失策出塁", "三振", "得点", "打率", "出塁率", "長打率"))
    rows = []
    for b in res["batting"]:
        rows.append("<tr>" + "".join(f"<td>{v}</td>" for v in (
            b["slot"], b["PA"], b["AB"], b["H"], b["doubles"], b["triples"], b["HR"], b["BB"], b["HBP"],
            b["ROE"], b["SO"], b["R"], fmt(b["AVG"]), fmt(b["OBP"]), fmt(b["SLG"]))) + "</tr>")
    return f"<div class=scroll><table><tr>{head}</tr>{''.join(rows)}</table></div>"


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


def result_page(res, source):
    body = (f"<p><a href='/'>← 戻る</a></p><h1>{html.escape(res['sheet'])}</h1>"
            f"<p class=mut>{html.escape(source)}・{res['columns']} 列</p>"
            "<h2>試合結果（回ごとの得点）</h2>" + line_score_html(res) +
            "<h2>確認してほしいところ</h2>" + checks_html(res) +
            "<h2>打順ごとの成績</h2>" + batting_html(res) +
            "<p class=mut>選手名は読んでいません（打順の番号で出します）。途中交代は同じ打順にまとめています。</p>"
            "<h2>読んだ格子（名前の欄は写していません）</h2>"
            f"<img alt='scoring grid' src='data:image/jpeg;base64,{res['image']}'>")
    return page(res["sheet"], body)


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
                  "<input type=file name=photo accept='image/*' required>"
                  "<label><input type=checkbox name=twice> 2 回読む（確認用・費用 2 倍）</label>"
                  "<button>読む</button></form>"
                  "<p id=busy class=warn hidden>読んでいます（20 秒ほど）…</p>"
                  "<p class=mut>1 枚あたり列 13 回＋得点表 1 回の API 呼び出し、20 秒ほど。返答は保存し、同じ写真は二度と払いません。</p>")
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

    def has_key(self):
        return bool(os.environ.get("SCOREBOOK_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))

    def do_GET(self):
        path = urllib.parse.unquote(self.path.split("?")[0])
        if path == "/":
            return self.send(200, index_page(self.has_key()))
        if path.startswith("/sheet/"):
            sheet = path[len("/sheet/"):]
            found = stored_sheets().get(sheet)
            if not found:
                return self.send(404, page("not found", "<p>その用紙はありません。</p>"))
            label, photo, fdir, f2dir, hdir = found
            reader = read.ReplayReader(None, None, facts_v2=fdir)
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
            return self.send(200, result_page(res, f"保存済みの返答を再生・{label}"))
        return self.send(404, page("not found", "<p>not found</p>"))

    def do_POST(self):
        if self.path != "/read":
            return self.send(404, page("not found", "<p>not found</p>"))
        if not self.has_key():
            return self.send(200, index_page(False, "API キーが無いので読めません。"))
        n = int(self.headers.get("Content-Length") or 0)
        if n > 25 * 1024 * 1024:
            return self.send(200, index_page(True, "写真が大きすぎます（25 MB まで）。"))
        msg = BytesParser(policy=email_policy).parsebytes(
            b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + self.rfile.read(n))
        photo_bytes, twice = None, False
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if name == "photo":
                photo_bytes = part.get_payload(decode=True)
            elif name == "twice":
                twice = True
        if not photo_bytes:
            return self.send(200, index_page(True, "写真がありません。"))
        sheet = "upload_" + hashlib.md5(photo_bytes).hexdigest()[:12]
        os.makedirs(UPLOADS, exist_ok=True)
        photo = os.path.join(UPLOADS, sheet + ".jpg")
        with open(photo, "wb") as f:
            f.write(photo_bytes)
        try:
            cols = grid.columns(photo)
        except (Exception, SystemExit) as e:  # unreadable file or no grid (cv/ raises SystemExit)
            why = str(e).replace(photo, "写真")
            return self.send(200, index_page(True, f"格子が見つかりませんでした（{why}）。用紙全体が入るように、正面から撮り直してください。"))
        if not 4 <= len(cols) <= 13:
            return self.send(200, index_page(True, f"回の列が {len(cols)} 本しか見つかりませんでした。撮り直してください。"))
        reader = read.ApiReader(CACHE)
        reader2 = read.ApiReader(CACHE + "-2") if twice else None
        try:
            res = analyse(photo, sheet, reader, reader2, reader)
        except RuntimeError as e:
            return self.send(200, index_page(True, f"API の呼び出しに失敗しました：{e}"))
        except (Exception, SystemExit) as e:
            return self.send(200, index_page(True, f"読めませんでした（{e}）。"))
        return self.send(200, result_page(res, "API で読み取り" + ("（2 回）" if twice else "")))

    def log_message(self, fmt_, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt_ % args))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m scorebook.web")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args(argv)
    print(f"http://{a.host}:{a.port}/", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
