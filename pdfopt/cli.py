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
    p.add_argument("input", nargs="+",
                   help="scanned book PDF(s) — multiple files or a shell "
                        "glob like *.pdf are converted one after another")
    p.add_argument("-o", "--output", help="output PDF path "
                   "(default: <input>_optimized.pdf; single input only)")
    p.add_argument("--pages", help="1-indexed subset like '5,10-12' (for testing)")
    p.add_argument("--dpi", type=int, default=300, help="processing DPI (default 300)")
    p.add_argument("--margin", type=float, default=0.05,
                   help="side margin as fraction of text width (default 0.05)")
    p.add_argument("--jpeg-quality", type=int, default=82, help="default 82")
    p.add_argument("--workers", type=int, help="parallel workers (default: cores-2)")
    p.add_argument("--comic", action="store_true",
                   help="comic-book mode: de-skew from panel-frame lines, keep "
                        "page borders (no column re-layout); all pages trimmed "
                        "to the book's smallest post-rotation size, cutting "
                        "from whichever side has more blank margin")
    p.add_argument("--no-ocr", action="store_true", help="skip the OCR text layer")
    p.add_argument("--ocr", action="store_true",
                   help="force the OCR layer on (comic mode has it off by default)")
    p.add_argument("--no-txt", action="store_true",
                   help="skip the paragraph-reflowed <input>.txt export")
    p.add_argument("--no-clean", action="store_true", help="keep original tone "
                   "(no show-through removal)")
    p.add_argument("--knee", default=None,
                   help="show-through knee 'k0,k1' in gray levels (default "
                        "150,205; comic mode: tone cleanup is OFF unless this "
                        "is given — screentone lives in the light grays)")
    p.add_argument("--no-cache", action="store_true",
                   help="ignore/skip the .analysis.json cache")
    args = p.parse_args(argv)

    inputs = []
    for f in dict.fromkeys(args.input):  # dedupe, keep order
        if f.lower().endswith("_optimized.pdf"):
            print(f"skip (이미 변환된 결과물): {f}", file=sys.stderr)
            continue
        inputs.append(f)
    if not inputs:
        p.error("no input files left to convert")
    if args.output and len(inputs) > 1:
        p.error("-o works with a single input file only")
    missing = [f for f in inputs if not os.path.exists(f)]
    if missing:
        p.error(f"not found: {missing[0]}")

    knee0, knee1 = 150, 205
    if args.knee is not None:
        try:
            parts = [int(v) for v in args.knee.split(",")]
            if len(parts) != 2:
                raise ValueError
            knee0, knee1 = parts
            if not (0 <= knee0 < knee1 <= 255):
                raise ValueError
        except ValueError:
            p.error(f"--knee must be 'k0,k1' with 0 <= k0 < k1 <= 255, "
                    f"got '{args.knee}'")

    if args.comic:
        # comics: tone cleanup only when explicitly requested, OCR opt-in,
        # no paragraph txt (speech bubbles don't reflow meaningfully)
        clean_tone = args.knee is not None and not args.no_clean
        use_ocr = args.ocr
        want_txt = False
    else:
        clean_tone = not args.no_clean
        use_ocr = not args.no_ocr
        want_txt = not args.no_txt

    try:
        pages = parse_pages(args.pages) if args.pages else None
    except ValueError as e:
        p.error(str(e))

    from .convert import convert
    failed, page_errors = [], False
    for n, src in enumerate(inputs, 1):
        if len(inputs) > 1:
            print(f"\n[{n}/{len(inputs)}] {os.path.basename(src)}")
        base, _ = os.path.splitext(src)
        out_path = args.output or base + "_optimized.pdf"
        cache = None if args.no_cache else base + ".analysis.json"
        try:
            res = convert(
                src, out_path,
                dpi=args.dpi, margin=args.margin, jpeg_quality=args.jpeg_quality,
                use_ocr=use_ocr, clean_tone=clean_tone,
                knee0=knee0, knee1=knee1, workers=args.workers,
                pages=pages, cache_path=cache, comic=args.comic,
                txt_path=os.path.join(
                    os.path.dirname(os.path.abspath(out_path)),
                    os.path.basename(base) + ".txt") if want_txt else None)
            page_errors = page_errors or bool(res["errors"])
        except (ValueError, RuntimeError) as e:
            print(f"error: {src}: {e}", file=sys.stderr)
            failed.append(src)
    if failed:
        print(f"\n{len(failed)}/{len(inputs)}권 실패: "
              f"{[os.path.basename(f) for f in failed]}", file=sys.stderr)
        return 2
    return 1 if page_errors else 0


if __name__ == "__main__":
    sys.exit(main())
