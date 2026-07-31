"""De-skew with per-line angle drift (keystone) correction.

Sheet-fed scanners often rotate the page progressively during the pass, so
each text line is straight but its angle drifts linearly from top to bottom.
A single rotation cannot fix that; instead we build a smooth displacement
field that rotates every line by its own measured angle:

    theta(y) = a + b*y                       (fitted over full-width lines)
    shear m  = d(column left edge x)/dy      (measured after flattening)

    inverse map (output -> source), applied as one cv2.remap:
        xi = xo + m * (yo - h/2)
        yi = yo + (xi - xc) * tan(theta(yo))
"""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import binarize_clean, detect_lines, split_regions

MAX_ABS_ANGLE_DEG = 3.0  # distrust larger estimates (art pages, noise)
MAX_SHEAR_DEG = 3.0
MIN_FULL_LINES = 4


def fit_theta(full_lines: list[dict]) -> tuple[float, float, float] | None:
    """Fit theta(y) = a + b*y (degrees, y in px). Returns (a, b, xc) or None."""
    pts = [(l["yc"], l["angle"], l["w"]) for l in full_lines if l["angle"] is not None]
    if len(pts) < MIN_FULL_LINES:
        return None
    ys = np.array([p[0] for p in pts])
    angs = np.array([p[1] for p in pts])
    wts = np.array([p[2] for p in pts], dtype=np.float64)

    if len(pts) >= 6 and np.ptp(ys) > 1:
        b, a = np.polyfit(ys, angs, 1, w=wts)
        # one robustness pass: drop >2.5 sigma outliers, refit
        resid = angs - (a + b * ys)
        sd = resid.std()
        if sd > 1e-6:
            keep = np.abs(resid) < 2.5 * sd
            if keep.sum() >= MIN_FULL_LINES and keep.sum() < len(pts):
                b, a = np.polyfit(ys[keep], angs[keep], 1, w=wts[keep])
    else:
        b, a = 0.0, float(np.average(angs, weights=wts))

    span = ys.max() - ys.min()
    if abs(a + b * ys.min()) > MAX_ABS_ANGLE_DEG or abs(a + b * ys.max()) > MAX_ABS_ANGLE_DEG:
        return None  # implausible fit; leave the page alone
    xc = float(np.median([(l["x0"] + l["x1"]) / 2 for l in full_lines]))
    return float(a), float(b), xc


def _measure_shear(binimg: np.ndarray, dpi: int) -> float:
    """Slope dx/dy of the column left edge on an already-flattened binary."""
    h, w = binimg.shape
    full = split_regions(detect_lines(binimg, dpi), w, h)["full"]
    if len(full) < MIN_FULL_LINES:
        return 0.0
    x0s = np.array([l["x0"] for l in full], dtype=np.float64)
    ys = np.array([l["yc"] for l in full], dtype=np.float64)
    med = np.median(x0s)
    keep = np.abs(x0s - med) < 0.06 * w  # exclude indented lines
    if keep.sum() < MIN_FULL_LINES or np.ptp(ys[keep]) < 1:
        return 0.0
    m, _ = np.polyfit(ys[keep], x0s[keep], 1)
    if abs(np.degrees(np.arctan(m))) > MAX_SHEAR_DEG:
        return 0.0
    return float(m)


def deskew(img: np.ndarray, dpi: int) -> tuple[np.ndarray, dict]:
    """Correct line-angle drift and column lean. Returns (corrected, info)."""
    binimg = binarize_clean(img)
    h, w = img.shape
    full = split_regions(detect_lines(binimg, dpi), w, h)["full"]
    fit = fit_theta(full)
    info = dict(applied=False, a=0.0, b=0.0, shear=0.0)
    if fit is None:
        return img, info
    a, b, xc = fit

    yo = np.arange(h, dtype=np.float32)[:, None]
    xo = np.arange(w, dtype=np.float32)[None, :]
    tan_theta = np.tan(np.radians(a + b * yo)).astype(np.float32)

    # flatten the binary (nearest) once, only to measure residual shear
    map_y = yo + (xo - xc) * tan_theta
    map_x = np.broadcast_to(xo, (h, w)).astype(np.float32).copy()
    flat_bin = cv2.remap(binimg, map_x, map_y, cv2.INTER_NEAREST)
    m = _measure_shear(flat_bin, dpi)

    # composed single-resample map: shear then flatten
    map_x = xo + np.float32(m) * (yo - h / 2.0)
    map_x = np.broadcast_to(map_x, (h, w)).astype(np.float32)
    map_y = yo + (map_x - xc) * tan_theta
    out = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    info.update(applied=True, a=a, b=b, shear=m)
    return out, info
