#!/usr/bin/env python3
"""Run grid detection over a directory of scans; one summary line each.

Usage: grid_batch.py <sample_dir>
"""
import cv2
import glob
import os
import sys

import grid_detect
import cell_diff


def main():
    sample_dir = sys.argv[1]
    paths = sorted(glob.glob(os.path.join(sample_dir, "*.jpg")))
    ok = 0
    for p in paths:
        name = os.path.basename(p)
        try:
            img = grid_detect.load_rotated(p)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            res = grid_detect.detect(gray)
            cols = cell_diff.inning_col_bounds(res["v_clusters"])
            rows = cell_diff.batter_row_bounds(res["h_clusters"])
            strip = cell_diff.strip_width(res["v_clusters"])
            status = "OK" if len(cols) == 13 and len(rows) == 11 else "PARTIAL"
            if status == "OK":
                ok += 1
            print(
                f"{name}: {status} cols={len(cols)} rows={len(rows)} "
                f"strip={strip} col0={cols[0][0]} row0={rows[0][0]}"
            )
        except SystemExit as e:
            print(f"{name}: FAIL ({e})")
        except Exception as e:  # keep the batch going
            print(f"{name}: ERROR ({type(e).__name__}: {e})")
    print(f"summary: {ok}/{len(paths)} OK")


if __name__ == "__main__":
    main()
