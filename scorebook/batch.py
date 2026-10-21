"""Read every column of every sheet in a manifest with the API, caching each reply.

Usage: python -m scorebook.batch <manifest> <cache_dir> [--kind facts_v2] [--sheets a,b] [--max-calls N]

Columns are cut by live grid detection (grid.columns). Replies already in
<cache_dir>/<kind>/ are skipped, so an interrupted run resumes for free.
Stops on the first 400/401/403/404 instead of burning calls.
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading

from . import grid, read


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m scorebook.batch")
    ap.add_argument("manifest")
    ap.add_argument("cache_dir")
    ap.add_argument("--kind", default="facts_v2", choices=read.KINDS)
    ap.add_argument("--sheets", help="comma-separated sheet ids (default: all)")
    ap.add_argument("--max-calls", type=int, default=0, help="stop after this many new API calls (0 = no limit)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)

    reader = read.ApiReader(a.cache_dir)
    keep = set(a.sheets.split(",")) if a.sheets else None
    jobs = []
    for g in json.load(open(a.manifest)):
        sheet = os.path.basename(g["gt"])[:-5]
        if keep and sheet not in keep:
            continue
        # header: one crop per sheet (the line-score box), saved as <sheet>_ch.json
        crops = [grid.Column("h", grid.header(g["image"]), 0)] if a.kind == "header" else grid.columns(g["image"])
        for c in crops:
            done = os.path.exists(os.path.join(a.cache_dir, a.kind, f"{sheet}_c{c.index}.json"))
            jobs.append((sheet, c, done))
    todo = [j for j in jobs if not j[2]]
    if a.max_calls:
        todo = todo[:a.max_calls]
    print(f"{len(jobs)} columns, {len(jobs) - len([j for j in jobs if not j[2]])} cached, calling {len(todo)}", flush=True)

    lock = threading.Lock()
    bad = []

    def one(job):
        sheet, c, _ = job
        cells = reader.read(a.kind, sheet, c)
        with lock:
            if cells is None:
                bad.append(f"{sheet}_c{c.index}")
            print(f"{sheet}_c{c.index}: {'ok' if cells is not None else 'UNPARSED'}", flush=True)

    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one, j) for j in todo]
        for f in cf.as_completed(futs):
            try:
                f.result()
            except RuntimeError as e:  # fatal API error: stop everything
                print(f"FATAL {e}", flush=True)
                os._exit(2)

    tok = {k: sum(u.get(k) or 0 for u in reader.usage) for k in
           ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")}
    print(f"calls {len(reader.usage)}  tokens {tok}  unparsed {len(bad)} {bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
