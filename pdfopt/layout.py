"""Global layout: derive the uniform canvas from whole-book statistics and
place each corrected page onto it.

Horizontal: canvas width = max body column width + 5% margin each side; each
page's text column is centered (illustration pages center by ink bbox).
Vertical: the book-wide common empty top/bottom bands are cropped away — every
page gets the same dy shift, so in-book vertical rhythm is preserved.
"""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import detect_lines, ink_bbox, split_regions


def global_stats(records: dict[int, dict], margin_frac: float = 0.05) -> dict:
    """Compute canvas geometry (points) from per-page analysis records."""
    text = [r for r in records.values()
            if r.get("kind") == "text" and r.get("n_full", 0) >= 5]
    if not text:
        raise ValueError("no text pages found — is this a scanned book PDF?")
    widths = np.array([r["col_width"] for r in text])
    # max width, but guard against a single merged-blob outlier
    cap = float(np.percentile(widths, 99.5)) + 2.0
    text_w = float(min(widths.max(), cap))

    tops, bottoms = [], []
    for r in records.values():
        if r.get("ink_bbox"):
            tops.append(r["ink_bbox"][1])
            bottoms.append(r["ink_bbox"][3])
    top_g = float(np.percentile(tops, 1))
    bottom_g = float(np.percentile(bottoms, 99))

    margin = margin_frac * text_w
    return dict(
        text_width=round(text_w, 2),
        margin=round(margin, 2),
        canvas_w=round(text_w + 2 * margin, 2),
        content_top=round(top_g, 2),
        content_bottom=round(bottom_g, 2),
        canvas_h=round((bottom_g - top_g) + 2 * margin, 2),
    )


def canvas_px(gs: dict, dpi: int) -> tuple[int, int]:
    pt2px = dpi / 72.0
    return int(round(gs["canvas_w"] * pt2px)), int(round(gs["canvas_h"] * pt2px))


def place(img: np.ndarray, line_bin: np.ndarray, ink_bin: np.ndarray,
          dpi: int, gs: dict) -> np.ndarray:
    """Paste a corrected page image onto the uniform canvas.

    line_bin: cleaned binary (for column detection); ink_bin: the binary that
    defines what counts as ink and must never be cropped (raw when cleaning
    removed suspiciously much, e.g. full-bleed artwork touching page edges).
    """
    h, w = img.shape
    pt2px = dpi / 72.0
    cw, ch = canvas_px(gs, dpi)
    canvas = np.full((ch, cw), 255, np.uint8)

    full = split_regions(detect_lines(line_bin, dpi), w, h)["full"]
    bbox = ink_bbox(ink_bin)
    if bbox is None:
        return canvas  # blank page stays blank
    bx0, by0, bx1, by1 = bbox

    if len(full) >= 3:
        col_l = float(np.median([l["x0"] for l in full]))
        col_r = float(np.median([l["x1"] for l in full]))
    else:
        col_l, col_r = float(bx0), float(bx1)

    # content larger than the canvas (oversized art): scale to fit, center it
    scale = min(1.0,
                (cw - 4) / max(1, bx1 - bx0),
                (ch - 4) / max(1, by1 - by0))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = img.shape
        bx0, by0, bx1, by1 = [int(v * scale) for v in (bx0, by0, bx1, by1)]
        dx = int(round(cw / 2.0 - (bx0 + bx1) / 2.0))
        dy = int(round(ch / 2.0 - (by0 + by1) / 2.0))
    else:
        dx = int(round(cw / 2.0 - (col_l + col_r) / 2.0))
        dy = int(round((gs["margin"] - gs["content_top"]) * pt2px))
    # never let the shift crop actual ink
    dx = min(max(dx, -bx0), (cw - 1) - bx1)
    dy = min(max(dy, -by0), (ch - 1) - by1)

    sx0, dx0 = max(0, -dx), max(0, dx)
    sy0, dy0 = max(0, -dy), max(0, dy)
    ww = min(w - sx0, cw - dx0)
    hh = min(h - sy0, ch - dy0)
    if ww > 0 and hh > 0:
        canvas[dy0 : dy0 + hh, dx0 : dx0 + ww] = img[sy0 : sy0 + hh, sx0 : sx0 + ww]
    return canvas
