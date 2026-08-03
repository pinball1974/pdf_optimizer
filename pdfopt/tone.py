"""Tone cleanup: suppress show-through while keeping glyph anti-aliasing.

A soft knee maps near-background grays to white. Real ink on these scans sits
around 60-130; paper is ~225-250; show-through from the reverse side lands in
between (~170-215), so the default knee removes it without eroding glyphs.
"""
from __future__ import annotations

import cv2
import numpy as np


def knee_lut(knee0: int = 150, knee1: int = 205) -> np.ndarray:
    lut = np.arange(256, dtype=np.float64)
    t = np.clip((lut - knee0) / max(1, knee1 - knee0), 0.0, 1.0)
    lut = lut * (1 - t) + 255.0 * t
    return lut.astype(np.uint8)


def clean(img: np.ndarray, knee0: int = 150, knee1: int = 205) -> np.ndarray:
    """Suppress show-through, adapting the knee to the page's ink darkness.

    Lightly-printed scans have glyph strokes well above the default knee0
    (e.g. median ink gray ~170 vs the assumed <150), and a fixed knee would
    erode them. The knee is therefore only ever RAISED: at least 40 levels
    above the page's median ink gray, never lowered below the caller's value.
    """
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = img[mask > 0]
    if ink.size >= 500:
        k0 = int(max(knee0, min(230, float(np.median(ink)) + 40)))
        knee1 = min(255, max(knee1, k0 + 50))
        knee0 = k0
    return cv2.LUT(img, knee_lut(knee0, knee1))
