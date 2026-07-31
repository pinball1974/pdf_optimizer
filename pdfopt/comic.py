"""Comic-book mode: de-skew without disturbing the page borders.

Comics have no text lines to measure, so skew comes from the long straight
panel-frame lines (Hough segments near 0/90 deg). Rotation forces an inward
crop (the axis-aligned rectangle inscribed in the rotated page); its size
varies with each page's angle, so pass 1 computes every page's inscribed
size, the smallest width/height across the book become the uniform target,
and each page's excess is trimmed from whichever side has more blank margin —
artwork that runs to an edge is never the first thing cut.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .geometry import binarize_raw

# beyond this the measurement is distrusted: warn and apply the cap only,
# so one bad page cannot shrink the whole book's target size
MAX_COMIC_ANGLE = 1.5


def _refine_dev(edges: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float | None:
    """Sub-pixel deviation of one Hough segment from 0/90 deg.

    Hough endpoints are integer pixels, so a 0.3 deg tilt over a short segment
    quantizes to exactly 0 — instead fit ALL edge pixels in a narrow corridor
    along the segment; hundreds of points give sub-pixel slope precision.
    """
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 8:
        return None
    pad = 4
    xa, xb = sorted((x1, x2))
    ya, yb = sorted((y1, y2))
    roi = edges[max(0, ya - pad) : yb + pad + 1, max(0, xa - pad) : xb + pad + 1]
    ys, xs = np.nonzero(roi)
    if len(xs) < 0.5 * length:
        return None
    gx = xs + max(0, xa - pad) - x1
    gy = ys + max(0, ya - pad) - y1
    dist = (gx * dy - gy * dx) / length  # signed distance to the segment line
    m = np.abs(dist) <= 2.5
    if m.sum() < 0.5 * length:
        return None
    gx, gy = gx[m].astype(np.float64), gy[m].astype(np.float64)
    if abs(dx) >= abs(dy):  # near-horizontal: fit y over x
        if np.ptp(gx) < 8:
            return None
        slope = np.polyfit(gx, gy, 1)[0]
        return float(math.degrees(math.atan(slope)))
    if np.ptp(gy) < 8:  # near-vertical: fit x over y
        return None
    slope = np.polyfit(gy, gx, 1)[0]
    return float(-math.degrees(math.atan(slope)))


def estimate_angle(img: np.ndarray, dpi: int) -> tuple[float, float]:
    """Skew angle from long near-horizontal/vertical lines (panel frames).

    Returns (angle_deg, support_px). angle > 0 = frame lines descend to the
    right (y grows downward). support is the total length of usable segments;
    when it is too small the angle is 0.0 (page left untouched).
    """
    edges = cv2.Canny(img, 60, 160)
    min_len = int(0.15 * min(img.shape))
    segs = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=100,
                           minLineLength=min_len,
                           maxLineGap=max(3, int(dpi * 0.03)))
    if segs is None:
        return 0.0, 0.0
    devs, lens, ycs = [], [], []
    for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4):
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        coarse = ((ang + 45.0) % 90.0) - 45.0  # deviation from nearest 0/90
        if abs(coarse) > 4.0:
            continue
        dev = _refine_dev(edges, int(x1), int(y1), int(x2), int(y2))
        if dev is None or abs(dev) > 4.0:
            dev = coarse
        devs.append(dev)
        lens.append(math.hypot(x2 - x1, y2 - y1))
        ycs.append((y1 + y2) / 2.0)
    if not devs:
        return 0.0, 0.0
    support = float(sum(lens))
    if len(devs) < 3 or support < 1.2 * min(img.shape):
        return 0.0, support
    d = np.asarray(devs)
    wl = np.asarray(lens, dtype=np.float64)
    med = _weighted_median(d, wl)
    # consensus check: panel frames all share the page's skew (tiny spread),
    # while directional artwork (speed/focus lines near 0 or 90 deg) shows a
    # wide spread — trusting its median would rotate a straight page
    mad = _weighted_median(np.abs(d - med), wl)
    if mad > 0.6:
        return 0.0, support
    # scanner feed drift makes the angle vary with y; a single rotation can't
    # remove the drift, but anchoring it at the PAGE CENTER splits the residual
    # evenly between top and bottom instead of dumping it all on one edge
    ys = np.asarray(ycs, dtype=np.float64)
    keep = np.abs(d - med) < 1.5
    if keep.sum() >= 6 and np.ptp(ys[keep]) > 0.25 * img.shape[0]:
        b, a = np.polyfit(ys[keep], d[keep], 1, w=wl[keep])
        center = a + b * (img.shape[0] / 2.0)
        if abs(center - med) < 0.75:  # sanity: fit must agree with the median
            return float(center), support
    return float(med), support


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, cum[-1] / 2.0)])


def inscribed_size(w: float, h: float, angle_deg: float) -> tuple[float, float]:
    """Largest axis-aligned rectangle inside a w x h rectangle rotated by
    angle_deg (the classic 'rotatedRectWithMaxArea' formula)."""
    a = abs(math.radians(angle_deg)) % math.pi
    if a > math.pi / 2:
        a = math.pi - a
    if a < 1e-8:
        return float(w), float(h)
    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)
    sin_a, cos_a = math.sin(a), math.cos(a)
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr = (w * cos_a - h * sin_a) / cos_2a
        hr = (h * cos_a - w * sin_a) / cos_2a
    return wr, hr


def deskew_crop(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate the page level and crop to the inscribed rectangle, so no
    invented white wedges remain at the corners."""
    h, w = img.shape
    if abs(angle_deg) > 1e-4:
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    iw, ih = inscribed_size(w, h, angle_deg)
    iw, ih = min(int(iw), w), min(int(ih), h)
    x0, y0 = (w - iw) // 2, (h - ih) // 2
    return img[y0 : y0 + ih, x0 : x0 + iw]


def edge_margins(img: np.ndarray) -> tuple[int, int, int, int]:
    """Blank margin (px) from each edge to the first real ink: (L, R, T, B).
    A thin band at the very edge is ignored so cut-edge scanner shadows do
    not count as artwork."""
    b = binarize_raw(img)
    e = max(3, int(0.004 * max(b.shape)))
    b[:e, :] = 0
    b[-e:, :] = 0
    b[:, :e] = 0
    b[:, -e:] = 0
    h, w = b.shape
    xs = np.where(np.count_nonzero(b, axis=0) > 2)[0]
    ys = np.where(np.count_nonzero(b, axis=1) > 2)[0]
    if len(xs) == 0 or len(ys) == 0:
        return w // 2, w // 2, h // 2, h // 2  # blank page: everything is margin
    return int(xs[0]), int(w - 1 - xs[-1]), int(ys[0]), int(h - 1 - ys[-1])


def allocate_trim(excess: int, m0: int, m1: int) -> tuple[int, int]:
    """Split `excess` px between two opposite sides with blank margins m0/m1.
    Prefer the side with more margin (leaving the remaining margins as equal
    as possible); cut into ink only when both margins are exhausted, and then
    split the damage evenly."""
    if excess <= 0:
        return 0, 0
    lo, hi = max(0, excess - m1), min(excess, m0)
    if lo <= hi:  # the excess fits inside the margins
        t0 = min(max(int(round((excess + m0 - m1) / 2.0)), lo), hi)
    else:
        t0 = int(round(m0 + (excess - m0 - m1) / 2.0))
        t0 = max(0, min(excess, t0))
    return t0, excess - t0


def comic_stats(records: dict[int, dict]) -> dict:
    """Uniform target size = smallest inscribed width/height in the book."""
    pages = [r for r in records.values() if r.get("kind") == "comic"]
    if not pages:
        raise ValueError("no analyzable pages found for comic mode")
    ws = np.array([r["insc_w"] for r in pages])
    hs = np.array([r["insc_h"] for r in pages])
    return dict(
        target_w=round(float(ws.min()), 2),
        target_h=round(float(hs.min()), 2),
        median_w=round(float(np.median(ws)), 2),
        median_h=round(float(np.median(hs)), 2),
        clamped=sorted(i + 1 for i, r in records.items()
                       if r.get("kind") == "comic" and r.get("clamped")),
        no_lines=sum(1 for r in pages if r.get("support", 0) == 0),
    )
