"""Find the scoring grid on a photo and cut one image per inning column."""
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cv"))
import cell_diff  # noqa: E402
import grid_detect  # noqa: E402

MARGIN = 12  # same margin as experiments/claude_vision_2026-10/crops.py


class Column:
    def __init__(self, index, png, nrows):
        self.index = index
        self.png = png  # encoded PNG bytes
        self.nrows = nrows


def detect(image_path):
    """Rotate/deskew the photo and find inning columns and batter rows."""
    img = grid_detect.load_rotated(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = grid_detect.detect(gray)
    res.pop("_masks", None)
    cols, rows = cell_diff.lattice(res)
    return img, cols, rows


def columns(image_path):
    """One crop per inning column: printed header row + all batter rows."""
    img, cols, rows = detect(image_path)
    H, W = img.shape[:2]
    y0, y1 = max(0, rows[0][0] - MARGIN), min(H, rows[-1][1] + MARGIN)
    out = []
    for ci, (x0, x1) in enumerate(cols):
        crop = img[y0:y1, max(0, x0 - MARGIN):min(W, x1 + MARGIN)]
        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            raise RuntimeError(f"could not encode column {ci}")
        out.append(Column(ci, buf.tobytes(), len(rows) - 1))
    return out


def header(image_path):
    """The line-score box above the grid (both teams' runs per inning), as PNG bytes."""
    img, cols, rows = detect(image_path)
    H, W = img.shape[:2]
    crop = img[0:min(H, rows[0][0] + MARGIN), 0:min(W, cols[-1][1])]
    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        raise RuntimeError("could not encode header")
    return buf.tobytes()
