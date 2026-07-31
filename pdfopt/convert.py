"""Conversion pipeline: analyze -> global layout -> per-page transform -> PDF.

Pass 1 (analysis, 150 dpi) records per-page geometry; global statistics derive
the uniform canvas. Pass 2 (300 dpi by default) cleans tone, de-skews, places
each page on the canvas, JPEG-encodes it, and optionally OCRs it. Pages are
processed in parallel worker processes; the PDF is assembled in order.

Order matters in pass 2: the tone knee runs BEFORE the de-skew resample —
bilinear remapping blends thin strokes with paper, and a knee applied after
would bleach them.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, as_completed

import cv2
import fitz
import numpy as np

from . import ocr as ocr_mod
from .deskew import deskew
from .geometry import binarize_clean, binarize_raw, page_metrics, render_gray
from .layout import canvas_px, global_stats, place
from .text_export import reflow
from .tone import clean

# gentler knee for non-text pages: pencil shading in illustrations lives in the
# midtones that the text knee would posterize
ART_KNEE = (190, 235)

_worker = {}


def _init_worker(pdf_path, settings):
    _worker["doc"] = fitz.open(pdf_path)
    _worker["settings"] = settings


def _analyze_one(idx):
    try:
        return idx, page_metrics(_worker["doc"][idx], dpi=150)
    except Exception as e:
        return idx, dict(kind="error", error=str(e))


def _convert_one(idx):
    s = _worker["settings"]
    dpi = s["dpi"]
    try:
        img = render_gray(_worker["doc"][idx], dpi)
        kind = s["kinds"].get(str(idx), "text")
        if s["clean_tone"]:
            if kind == "text":
                img = clean(img, s["knee0"], s["knee1"])
            else:
                img = clean(img, max(s["knee0"], ART_KNEE[0]),
                            max(s["knee1"], ART_KNEE[1]))
        img, info = deskew(img, dpi)
        line_bin = binarize_clean(img)
        raw_bin = binarize_raw(img)
        # if cleaning removed most of the ink, this is full-bleed artwork whose
        # edge-touching mass was mistaken for scanner junk — trust the raw map
        ink_bin = line_bin
        raw_ink = int(np.count_nonzero(raw_bin))
        if raw_ink and np.count_nonzero(line_bin) < 0.2 * raw_ink:
            ink_bin = raw_bin
        canvas = place(img, line_bin, ink_bin, dpi, s["gs"])
        ocr_lines = ocr_mod.recognize(canvas) if s["ocr"] else []
        if ocr_lines:
            ocr_lines = _drop_repeats(ocr_lines, canvas.shape)
        quality = s["jpeg_quality"] if kind == "text" else min(95, s["jpeg_quality"] + 10)
        ok, jpg = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("jpeg encode failed")
        return idx, dict(jpg=jpg.tobytes(), ocr=ocr_lines, deskew=info,
                         shape=canvas.shape)
    except Exception as e:
        return idx, dict(error=str(e))


def _drop_repeats(lines, shape):
    """Drop running headers and folio numbers from the OCR of a body-text
    page: short lines in the outer vertical bands are page furniture, not
    content (wide footnote blocks near the bottom survive the width test)."""
    ch, cw = shape
    out = []
    for ln in lines:
        yc = (ln["y0"] + ln["y1"]) / 2
        small = (ln["x1"] - ln["x0"]) < 0.25 * cw
        if small and (yc < 0.12 * ch or yc > 0.88 * ch):
            continue
        out.append(ln)
    return out


def _run_pool(pdf_path, settings, indices, fn, workers, label):
    t0 = time.time()
    results = {}
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(pdf_path, settings)) as ex:
            futures = [ex.submit(fn, i) for i in indices]
            for n, fut in enumerate(as_completed(futures), 1):
                idx, rec = fut.result()
                results[idx] = rec
                if n % 20 == 0 or n == len(indices):
                    el = time.time() - t0
                    print(f"  {label}: {n}/{len(indices)} ({el:.0f}s)", flush=True)
    except BrokenExecutor as e:
        raise RuntimeError(
            f"{label}: a worker process died (crash or out-of-memory) — "
            f"try fewer workers via --workers") from e
    return results


def analyze(pdf_path: str, cache_path: str | None, workers: int) -> dict[int, dict]:
    src_mtime = os.path.getmtime(pdf_path)
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            if cached.get("_mtime") == src_mtime:
                print(f"analysis cache hit: {cache_path}")
                return {int(k): v for k, v in cached["pages"].items()}
        except (json.JSONDecodeError, KeyError):
            pass
    doc = fitz.open(pdf_path)
    n = doc.page_count
    doc.close()
    print(f"pass 1/2: analyzing {n} pages ...")
    records = _run_pool(pdf_path, {}, list(range(n)), _analyze_one,
                        workers, "analyze")
    bad = sorted(i for i, r in records.items() if r.get("kind") == "error")
    if bad:
        print(f"WARNING: analysis failed on {len(bad)} page(s): "
              f"{[i + 1 for i in bad[:10]]}{'...' if len(bad) > 10 else ''} — "
              f"they are excluded from layout statistics", file=sys.stderr)
    if cache_path and not bad:
        # mtime captured BEFORE the pass: if the file was replaced meanwhile,
        # the next run sees a mismatch instead of reusing stale geometry
        with open(cache_path, "w") as f:
            json.dump({"_mtime": src_mtime,
                       "pages": {str(k): v for k, v in records.items()}}, f)
    return records


_korea_font = None


def _fit_fontsize(font, text: str, rect: "fitz.Rect") -> float:
    """Fontsize whose natural advance matches the OCR bbox width, so search
    highlights and tap-to-select stay aligned along the whole word."""
    adv = font.text_length(text, fontsize=1.0)
    if adv <= 0:
        return max(4.0, rect.height * 0.85)
    return max(2.0, rect.width / adv)


def _embed_ocr(page, lines, dpi):
    """One TextWriter per page: words appended at exact positions land in a
    single text object, so extraction merges same-baseline words into lines
    and multi-word phrase search works."""
    global _korea_font
    if _korea_font is None:
        _korea_font = fitz.Font("korea")
    px2pt = 72.0 / dpi
    tw = fitz.TextWriter(page.rect)
    for ln in lines:
        for l in ln["words"]:
            r = fitz.Rect(l["x0"] * px2pt, l["y0"] * px2pt,
                          l["x1"] * px2pt, l["y1"] * px2pt)
            try:
                # trailing space -> extraction sees real word boundaries, so
                # multi-word phrase search works; the next word is positioned
                # absolutely so the extra advance never shifts anything visible
                tw.append(fitz.Point(r.x0, r.y1 - 0.2 * r.height),
                          l["text"] + " ", font=_korea_font,
                          fontsize=_fit_fontsize(_korea_font, l["text"], r))
            except Exception:
                pass  # a failed OCR word must never break the page
    try:
        tw.write_text(page, render_mode=3)
    except Exception:
        pass


def convert(pdf_path: str, out_path: str, dpi: int = 300, margin: float = 0.05,
            jpeg_quality: int = 82, use_ocr: bool = True, clean_tone: bool = True,
            knee0: int = 150, knee1: int = 205, workers: int | None = None,
            pages: list[int] | None = None, cache_path: str | None = None,
            txt_path: str | None = None) -> dict:
    workers = workers or max(1, (os.cpu_count() or 4) - 2)
    doc = fitz.open(pdf_path)
    n = doc.page_count
    doc.close()
    if pages is not None:
        if not pages:
            raise ValueError("empty page selection")
        outside = [i for i in pages if i < 0 or i >= n]
        if outside:
            raise ValueError(
                f"page(s) out of range 1-{n}: {[i + 1 for i in outside]}")
    indices = pages if pages is not None else list(range(n))

    records = analyze(pdf_path, cache_path, workers)
    gs = global_stats(records, margin)
    print(f"canvas: {gs['canvas_w']:.1f} x {gs['canvas_h']:.1f} pt "
          f"(text width {gs['text_width']:.1f} pt, margin {gs['margin']:.1f} pt)")

    if use_ocr and not ocr_mod.AVAILABLE:
        print("WARNING: pyobjc Vision not available -> OCR layer skipped",
              file=sys.stderr)
        use_ocr = False

    settings = dict(dpi=dpi, gs=gs, jpeg_quality=jpeg_quality, ocr=use_ocr,
                    clean_tone=clean_tone, knee0=knee0, knee1=knee1,
                    kinds={str(i): r.get("kind", "text")
                           for i, r in records.items()})
    print(f"pass 2/2: converting {len(indices)} pages "
          f"(dpi {dpi}, {workers} workers, ocr {'on' if use_ocr else 'off'}) ...")
    results = _run_pool(pdf_path, settings, indices, _convert_one,
                        workers, "convert")

    # blank fallback pages must get the same px-quantized size as real ones
    cpx_w, cpx_h = canvas_px(gs, dpi)
    blank_w, blank_h = cpx_w * 72.0 / dpi, cpx_h * 72.0 / dpi

    out = fitz.open()
    errors = []
    pages_lines = []
    for idx in indices:
        rec = results.pop(idx, {})
        pages_lines.append(rec.get("ocr") or [])
        if "error" in rec or "jpg" not in rec:
            errors.append((idx, rec.get("error", "missing result")))
            out.new_page(width=blank_w, height=blank_h)
            continue
        ph, pw = rec["shape"]
        page = out.new_page(width=pw * 72.0 / dpi, height=ph * 72.0 / dpi)
        page.insert_image(page.rect, stream=rec["jpg"])
        if rec["ocr"]:
            _embed_ocr(page, rec["ocr"], dpi)
    out.set_metadata({"title": os.path.splitext(os.path.basename(pdf_path))[0],
                      "creator": "pdfopt"})
    out.save(out_path, deflate=True, garbage=3)
    out.close()
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"wrote {out_path} ({size_mb:.1f} MB)")
    if txt_path and use_ocr:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(reflow(pages_lines))
        print(f"wrote {txt_path}")
    for idx, err in errors:
        print(f"WARNING: page {idx + 1} failed: {err}", file=sys.stderr)
    return dict(pages=len(indices), errors=errors, canvas=gs, out=out_path)
