"""Claude readers for one inning column.

Two questions are asked per column, with the prompts measured in
experiments/claude_facts_2026-10 (facts) and experiments/claude_vision_2026-10
(corners). The request layouts are kept the same as in those runs so the
measured accuracy carries over.

- facts   : per cell, was there a plate appearance, how did it end, which out numeral
- corners : per cell, the marks at the 2B / 3B / home corners of the diamond
"""
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("SB_MODEL", "claude-opus-5-5")
MAX_TOKENS = 16000
API_URL = "https://api.anthropic.com/v1/messages"

FACTS_PROMPT = open(os.path.join(HERE, "prompts", "facts.txt"), encoding="utf-8").read()
CORNERS_PROMPT = open(os.path.join(HERE, "prompts", "corners.txt"), encoding="utf-8").read()
FACTS_V2_PROMPT = open(os.path.join(HERE, "prompts", "facts_v2.txt"), encoding="utf-8").read()
HEADER_PROMPT = open(os.path.join(HERE, "prompts", "header.txt"), encoding="utf-8").read()

KINDS = ("facts", "facts_v2", "corners", "header")


def facts_request(png, nrows, prompt=FACTS_PROMPT):
    img = base64.b64encode(png).decode()
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "thinking": {"type": "adaptive"},
        "system": [{"type": "text", "text": prompt.replace("{nrows}", "N"),
                    "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}},
            {"type": "text", "text": "This column has N = %d batter rows (r1 .. r%d)." % (nrows, nrows)},
        ]}],
    }


def facts_v2_request(png, nrows):
    """facts plus strikeout / scored / left_on_base (2026-10-21); same layout as facts."""
    return facts_request(png, nrows, FACTS_V2_PROMPT)


def header_request(png, nrows=0):
    """The line-score box at the top of the sheet (one call per sheet, 2026-10-21)."""
    img = base64.b64encode(png).decode()
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "thinking": {"type": "adaptive"},
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}},
            {"type": "text", "text": HEADER_PROMPT},
        ]}],
    }


REQUESTS = {"facts": facts_request, "facts_v2": facts_v2_request, "header": header_request}


def corners_request(png, nrows):
    img = base64.b64encode(png).decode()
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "thinking": {"type": "adaptive"},
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}},
            {"type": "text", "text": CORNERS_PROMPT.replace("{nrows}", str(nrows))},
        ]}],
    }


def reply_json(resp):
    """The JSON object in the last text block of an API response; None if unusable."""
    texts = [b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text"]
    if not texts:
        return None
    text = texts[-1].strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def parse_reply(resp):
    """Pull the {"cells": ...} object out of an API response; None if unusable."""
    obj = reply_json(resp)
    cells = obj.get("cells") if obj else None
    return cells if isinstance(cells, dict) else None


def parse_header(resp):
    """[{"innings": [...], "total": n}, ...] from a header reply; None if unusable."""
    obj = reply_json(resp)
    rows = obj.get("rows") if obj else None
    return rows if isinstance(rows, list) else None


PARSERS = {"header": parse_header}


class ReplayReader:
    """Serve replies stored as <dir>/<sheet>_c<col>.json (raw API responses)."""

    def __init__(self, facts_dir, corners_dir, **more):
        self.dirs = {"facts": facts_dir, "corners": corners_dir, **more}
        self.missing = []

    def read(self, kind, sheet, column):
        path = os.path.join(self.dirs[kind], f"{sheet}_c{column.index}.json")
        if not os.path.exists(path):
            self.missing.append(path)
            return None
        with open(path, encoding="utf-8") as f:
            return PARSERS.get(kind, parse_reply)(json.load(f))


class ApiReader:
    """Call the Anthropic API. Every raw response is saved under cache_dir,
    so a rerun replays from disk instead of paying again."""

    def __init__(self, cache_dir, api_key=None):
        # cloud sessions do not pass ANTHROPIC_API_KEY through, so SCOREBOOK_API_KEY is read first
        self.key = api_key or os.environ.get("SCOREBOOK_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not self.key:
            raise RuntimeError("SCOREBOOK_API_KEY / ANTHROPIC_API_KEY is not set (use --replay to run from stored replies)")
        self.cache_dir = cache_dir
        for kind in KINDS:
            os.makedirs(os.path.join(cache_dir, kind), exist_ok=True)
        self.usage = []

    def read(self, kind, sheet, column):
        path = os.path.join(self.cache_dir, kind, f"{sheet}_c{column.index}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return PARSERS.get(kind, parse_reply)(json.load(f))
        build = REQUESTS.get(kind, corners_request)
        resp = self._post(build(column.png, column.nrows))
        resp["_sent"] = {"model": MODEL, "thinking": "adaptive"}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(resp, f, ensure_ascii=False)
        self.usage.append(resp.get("usage", {}))
        return PARSERS.get(kind, parse_reply)(resp)

    def _post(self, body):
        data = json.dumps(body).encode()
        for attempt in range(5):
            req = urllib.request.Request(API_URL, data=data, headers={
                "x-api-key": self.key, "anthropic-version": "2023-06-01",
                "content-type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:500]
                if e.code in (400, 401, 403, 404) or attempt == 4:
                    raise RuntimeError(f"API error {e.code}: {msg}") from None
            except (urllib.error.URLError, TimeoutError):
                if attempt == 4:
                    raise
            time.sleep(15 * (attempt + 1))
