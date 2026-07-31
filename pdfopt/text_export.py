"""Reflow OCR lines into paragraphs and export plain text.

Hard line breaks inside a paragraph are removed using the print's own
typography:
  - an indented first line starts a new paragraph (Korean books indent ~1em)
  - a line that ends short of the justified right edge ends its paragraph
  - a vertical gap much larger than the page's line pitch is a section break
  - paragraphs continue across page boundaries; a page with no text
    (illustration) flushes the open paragraph

Lines are joined WITHOUT a separator: justified Korean breaks lines mid-word,
so the fragments concatenate back into whole words. When a break happens to
fall exactly on a word boundary the inter-word space is lost — a known,
rare limitation.
"""
from __future__ import annotations

import re

import numpy as np


def _page_paragraph_breaks(lines: list[dict]) -> tuple[list[bool], list[bool]]:
    """For each line: does a paragraph break come BEFORE it, and is the line
    itself paragraph-FINAL (ends short of the right edge)?"""
    n = len(lines)
    if n == 0:
        return [], []
    widths = np.array([l["x1"] - l["x0"] for l in lines])
    ems = np.array([l["y1"] - l["y0"] for l in lines])
    em = float(np.median(ems))
    full = widths > 0.6 * widths.max()
    if full.sum() >= 3:
        col_l = float(np.median([l["x0"] for l, f in zip(lines, full) if f]))
        col_r = float(np.median([l["x1"] for l, f in zip(lines, full) if f]))
    else:
        # sparse page (chapter title etc.): every line stands alone
        return [True] * n, [True] * n

    pitches = [lines[i + 1]["y0"] - lines[i]["y0"] for i in range(n - 1)]
    pitch = float(np.median(pitches)) if pitches else em * 1.6

    breaks_before, is_final = [], []
    for i, l in enumerate(lines):
        indented = l["x0"] > col_l + 0.5 * em
        big_gap = i > 0 and (l["y0"] - lines[i - 1]["y0"]) > 1.6 * pitch
        breaks_before.append(indented or big_gap)
        # generous threshold: justified lines can measure slightly short, and
        # a genuinely new paragraph is caught by the next line's indentation
        is_final.append(l["x1"] < col_r - 1.5 * em)
    return breaks_before, is_final


def reflow(pages_lines: list[list[dict]]) -> str:
    """pages_lines: per page, the OCR line dicts (already header/footer
    filtered). Returns paragraph-reflowed plain text."""
    paras: list[str] = []
    buf = ""

    def flush():
        nonlocal buf
        s = buf.strip()
        # bare 1-3 digit "paragraphs" are stray folio numbers split off a TOC
        # entry or specks misread as a digit — never real prose
        if s and not re.fullmatch(r"\d{1,3}", s):
            paras.append(s)
        buf = ""

    for lines in pages_lines:
        # bare digit lines (stray folio split off a TOC entry, specks misread
        # as a digit) are dropped BEFORE break analysis so they cannot split
        # the surrounding paragraph
        lines = [l for l in lines if not re.fullmatch(r"\d{1,3}", l["text"].strip())]
        if not lines:
            # illustration/blank pages appear MID-essay in some books, with a
            # sentence continuing on the next text page — keep the paragraph
            # open; a real essay boundary is caught by the short final line
            # and the next paragraph's indentation
            continue
        breaks_before, is_final = _page_paragraph_breaks(lines)
        for i, l in enumerate(lines):
            if breaks_before[i]:
                flush()
            # justified Korean breaks mid-word, so join without a separator —
            # but a break right after closing punctuation was a word boundary
            if buf and buf[-1] in ".?!…\"'』」）)":
                buf += " "
            buf += l["text"].strip()
            if is_final[i]:
                flush()
    flush()
    return "\n\n".join(paras) + "\n"
