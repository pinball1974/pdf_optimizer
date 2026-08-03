"""Page geometry: rendering, binarization, text-line detection, per-page metrics.

All pixel work happens at a caller-chosen DPI; metrics are reported in PDF
points (1/72 inch) so they are comparable across passes run at different DPIs.
"""
from __future__ import annotations

import cv2
import fitz
import numpy as np

# Angle estimates beyond this are treated as measurement noise, not skew.
MAX_LINE_ANGLE_DEG = 4.0


def render_gray(page: "fitz.Page", dpi: int) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    return img.reshape(pix.height, pix.width).copy()


def binarize_raw(img: np.ndarray) -> np.ndarray:
    """Plain Otsu binarization (ink=255), no artifact removal. Use this as the
    record of actual ink when deciding what must never be cropped: the cleaned
    variant can delete full-bleed artwork that touches the page edges."""
    _, binimg = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binimg


def binarize_clean(img: np.ndarray) -> np.ndarray:
    """Otsu binarization (ink=255) with scanner edge artifacts removed."""
    _, binimg = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = img.shape
    b = max(3, int(0.008 * max(h, w)))
    binimg[:b, :] = 0
    binimg[-b:, :] = 0
    binimg[:, :b] = 0
    binimg[:, -b:] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binimg)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        touches = x <= b or y <= b or x + cw >= w - b or y + ch >= h - b
        if touches and (cw > 0.5 * w or ch > 0.5 * h or area > 0.02 * w * h):
            binimg[lab == i] = 0
    return binimg


def detect_lines(binimg: np.ndarray, dpi: int) -> list[dict]:
    """Merge glyphs into text-line blobs; estimate each blob's baseline angle."""
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(dpi * 0.15)), 3))
    merged = cv2.dilate(binimg, kern)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(merged)
    lines = []
    for i in range(1, n):
        x, y, cw, ch, _ = stats[i]
        if ch < dpi * 0.03 or ch > dpi * 0.45 or cw < dpi * 0.05:
            continue
        roi = binimg[y : y + ch, x : x + cw]
        cols = np.where(roi.any(axis=0))[0]
        if len(cols) < dpi * 0.04:
            continue
        ys = np.arange(ch, dtype=np.float64)
        colmass = roi[:, cols].astype(np.float64)
        centroids = (colmass * ys[:, None]).sum(axis=0) / colmass.sum(axis=0)
        angle = None
        if len(cols) >= 8:
            slope, _b = np.polyfit(cols.astype(np.float64), centroids, 1)
            a = float(np.degrees(np.arctan(slope)))
            if abs(a) < MAX_LINE_ANGLE_DEG:
                angle = a
        lines.append(
            dict(x0=x, y0=y, x1=x + cw, y1=y + ch, w=cw, h=ch,
                 yc=y + ch / 2.0, angle=angle)
        )
    return lines


def split_regions(lines: list[dict], w: int, h: int) -> dict:
    """Classify line blobs into body / header / footer; pick full-width body lines."""
    body, header, footer = [], [], []
    for l in lines:
        small = l["w"] < 0.25 * w
        if l["yc"] < 0.12 * h and small:
            header.append(l)
        elif l["yc"] > 0.88 * h and small:
            footer.append(l)
        else:
            body.append(l)
    full = []
    if body:
        wmax = max(l["w"] for l in body)
        full = [l for l in body if l["w"] > 0.7 * wmax]
    return dict(body=body, header=header, footer=footer, full=full)


def ink_bbox(binimg: np.ndarray, min_area_px: int = 20) -> tuple | None:
    """Bounding box of all ink, ignoring specks below min_area_px."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binimg)
    x0 = y0 = 10**9
    x1 = y1 = -1
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < min_area_px:
            continue
        x0, y0 = min(x0, x), min(y0, y)
        x1, y1 = max(x1, x + cw), max(y1, y + ch)
    if x1 < 0:
        return None
    return (x0, y0, x1, y1)


def content_bbox(binimg: np.ndarray, lines: list[dict], dpi: int) -> tuple | None:
    """Bounding box of REAL content: detected line boxes plus ink components
    near them. Isolated specks and scanner edge strips far from any text are
    excluded — letting them into the bbox skews placement (false oversized
    extent -> needless shrink / off-center clamping) and inflates the global
    crop band. Pages with no detected lines fall back to the ink bbox away
    from a thin edge band."""
    h, w = binimg.shape
    e = max(3, int(0.015 * max(h, w)))
    inner = binimg.copy()
    inner[:e, :] = 0
    inner[-e:, :] = 0
    inner[:, :e] = 0
    inner[:, -e:] = 0
    if not lines:
        return ink_bbox(inner, min_area_px=30)
    x0 = min(l["x0"] for l in lines)
    y0 = min(l["y0"] for l in lines)
    x1 = max(l["x1"] for l in lines)
    y1 = max(l["y1"] for l in lines)
    m = int(0.03 * max(h, w))
    ex0, ey0, ex1, ey1 = x0 - m, y0 - m, x1 + m, y1 + m
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(inner)
    for i in range(1, n):
        cx, cy, cw, ch, area = stats[i]
        if area < 30:
            continue
        if cx < ex1 and cx + cw > ex0 and cy < ey1 and cy + ch > ey0:
            x0, y0 = min(x0, cx), min(y0, cy)
            x1, y1 = max(x1, cx + cw), max(y1, cy + ch)
    return (x0, y0, x1, y1)


def page_metrics(page: "fitz.Page", dpi: int = 150) -> dict:
    """One analysis record per page. Lengths in PDF points."""
    img = render_gray(page, dpi)
    h, w = img.shape
    px2pt = 72.0 / dpi
    binimg = binarize_clean(img)
    lines = detect_lines(binimg, dpi)
    regions = split_regions(lines, w, h)
    full = regions["full"]

    rec = dict(
        page_w=page.rect.width,
        page_h=page.rect.height,
        n_lines=len(lines),
        n_body=len(regions["body"]),
        n_full=len(full),
        has_header=bool(regions["header"]),
        has_footer=bool(regions["footer"]),
    )
    bbox = content_bbox(binimg, lines, dpi)
    rec["ink_bbox"] = [round(v * px2pt, 2) for v in bbox] if bbox else None

    if len(full) >= 3:
        rec["kind"] = "text"
        rec["col_left"] = round(float(np.median([l["x0"] for l in full])) * px2pt, 2)
        rec["col_right"] = round(float(np.median([l["x1"] for l in full])) * px2pt, 2)
        rec["col_width"] = round(rec["col_right"] - rec["col_left"], 2)
        angs = [l["angle"] for l in full if l["angle"] is not None]
        rec["angle_med"] = round(float(np.median(angs)), 3) if angs else None
    elif lines:
        rec["kind"] = "sparse"
    else:
        rec["kind"] = "empty" if rec["ink_bbox"] else "blank"
    return rec
