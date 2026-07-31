"""Apple Vision OCR: recognize Korean text on the final page image so an
invisible, searchable text layer can be embedded in the output PDF.

Requires pyobjc-framework-Vision. All processing is local to the Mac.
Coordinates returned are in image pixels, origin top-left.
"""
from __future__ import annotations

import re

import cv2
import numpy as np

try:
    import objc
    import Quartz  # noqa: F401  (pulls in CoreGraphics for CGImage creation)
    import Vision
    from Foundation import NSData, NSMakeRange

    AVAILABLE = True
except ImportError:  # pragma: no cover - depends on host setup
    AVAILABLE = False

LANGUAGES = ["ko-KR", "en-US"]


def recognize(img: np.ndarray) -> list[dict]:
    """OCR a grayscale page image.

    Returns one dict per recognized text LINE, top-to-bottom:
      {text, x0, y0, x1, y1, words: [{text, x0, y0, x1, y1}, ...]}
    Word boxes are exact (Vision boundingBoxForRange) but share the line's
    vertical band so a PDF text extractor keeps them on one line.
    """
    if not AVAILABLE:
        return []
    # long-lived worker processes have no running autorelease pool; without one
    # the ObjC temporaries (CGImage/Vision buffers) leak ~2 MB per call
    with objc.autorelease_pool():
        return _recognize(img)


def _recognize(img: np.ndarray) -> list[dict]:
    h, w = img.shape
    ok, png = cv2.imencode(".png", img)
    if not ok:
        return []
    data = NSData.dataWithBytes_length_(png.tobytes(), len(png))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setRecognitionLanguages_(LANGUAGES)
    request.setUsesLanguageCorrection_(True)
    success, _err = handler.performRequests_error_([request], None)
    if not success:
        return []
    out = []
    for obs in request.results() or []:
        cands = obs.topCandidates_(1)
        if not cands or not len(cands):
            continue
        cand = cands[0]
        text = str(cand.string())
        if not text.strip():
            continue
        lb = obs.boundingBox()  # normalized, origin bottom-left
        line = dict(
            text=text,
            x0=lb.origin.x * w,
            y0=(1.0 - lb.origin.y - lb.size.height) * h,
            x1=(lb.origin.x + lb.size.width) * w,
            y1=(1.0 - lb.origin.y) * h,
            words=[],
        )
        try:
            for m in re.finditer(r"\S+", text):
                box, _err = cand.boundingBoxForRange_error_(
                    NSMakeRange(m.start(), m.end() - m.start()), None)
                if box is None:
                    raise ValueError("no box for range")
                bb = box.boundingBox()
                line["words"].append(dict(
                    text=m.group(),
                    x0=bb.origin.x * w,
                    y0=line["y0"],
                    x1=(bb.origin.x + bb.size.width) * w,
                    y1=line["y1"],
                ))
        except Exception:
            # fall back to the whole line as a single "word"
            line["words"] = [dict(text=text, x0=line["x0"], y0=line["y0"],
                                  x1=line["x1"], y1=line["y1"])]
        out.append(line)
    out.sort(key=lambda l: (l["y0"], l["x0"]))
    return out
