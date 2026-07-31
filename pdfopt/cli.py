"""Command line interface for pdfopt."""
from __future__ import annotations

import argparse
import os
import sys


def parse_pages(spec: str) -> list[int]:
    """'5,10-12' -> [4, 9, 10, 11] (user-facing numbers are 1-indexed)."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = (int(v) for v in part.split("-", 1))
            if a < 1 or b < a:
                raise ValueError(f"invalid page range '{part}' (1-indexed, low-high)")
            out.extend(range(a - 1, b))
        else:
            v = int(part)
            if v < 1:
                raise ValueError(f"invalid page number '{part}' (pages are 1-indexed)")
            out.append(v - 1)
    if not out:
        raise ValueError("empty page selection")
    return sorted(set(out))


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="pdfopt",
        description="Straighten and re-center scanned book PDFs for tablet "
                    "reading: per-line de-skew (handles top-vs-bottom angle "
                    "drift), uniform text-width canvas, show-through cleanup, "
                    "searchable Korean OCR layer (Apple Vision).")
    p.add_argument("input", help="scanned book PDF")
    p.add_argument("-o", "--output", help="output PDF path "
                   "(default: <input>_optimized.pdf)")
    p.add_argument("--pages", help="1-indexed subset like '5,10-12' (for testing)")
    p.add_argument("--dpi", type=int, default=300, help="processing DPI (default 300)")
    p.add_argument("--margin", type=float, default=0.05,
                   help="side margin as fraction of text width (default 0.05)")
    p.add_argument("--jpeg-quality", type=int, default=82, help="default 82")
    p.add_argument("--workers", type=int, help="parallel workers (default: cores-2)")
    p.add_argument("--no-ocr", action="store_true", help="skip the OCR text layer")
    p.add_argument("--no-txt", action="store_true",
                   help="skip the paragraph-reflowed <input>.txt export")
    p.add_argument("--no-clean", action="store_true", help="keep original tone "
                   "(no show-through removal)")
    p.add_argument("--knee", default="150,205",
                   help="show-through knee 'k0,k1' in gray levels (default 150,205)")
    p.add_argument("--no-cache", action="store_true",
                   help="ignore/skip the .analysis.json cache")
    args = p.parse_args(argv)

    if not os.path.exists(args.input):
        p.error(f"not found: {args.input}")
    base, _ = os.path.splitext(args.input)
    out_path = args.output or base + "_optimized.pdf"
    cache = None if args.no_cache else base + ".analysis.json"

    try:
        parts = [int(v) for v in args.knee.split(",")]
        if len(parts) != 2:
            raise ValueError
        knee0, knee1 = parts
        if not (0 <= knee0 < knee1 <= 255):
            raise ValueError
    except ValueError:
        p.error(f"--knee must be 'k0,k1' with 0 <= k0 < k1 <= 255, got '{args.knee}'")

    try:
        pages = parse_pages(args.pages) if args.pages else None
    except ValueError as e:
        p.error(str(e))

    from .convert import convert
    try:
        res = convert(
            args.input, out_path,
            dpi=args.dpi, margin=args.margin, jpeg_quality=args.jpeg_quality,
            use_ocr=not args.no_ocr, clean_tone=not args.no_clean,
            knee0=knee0, knee1=knee1, workers=args.workers,
            pages=pages, cache_path=cache,
            txt_path=None if args.no_txt else os.path.join(
                os.path.dirname(os.path.abspath(out_path)),
                os.path.basename(base) + ".txt"))
    except (ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 1 if res["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
