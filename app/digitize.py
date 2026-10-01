"""Digitize HVO tilt plot PNGs (900x300 MATLAB-style plots) into time series.

USGS does not publish live UWD tilt as data (only 60-day-latency CSV releases), but it
does publish PNG plots refreshed every ~10 minutes. We recover the series by:

1. locating the black axes frame,
2. finding the y tick rows (tick marks on the frame) and OCR-ing their labels
   (tesseract) to build a robust linear pixel->µrad mapping,
3. reading the x range from the plot subtitle "(YYYY-MM-DD HH:MM:SS to ...)" via OCR,
   falling back to a caller-supplied (start, end),
4. extracting the blue (az 300) trace column by column: median, min and max row.

Resolution is limited by the image (≈0.02–0.7 µrad per pixel depending on the plot).
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from PIL import Image

from .config import HST

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None


@dataclass
class Digitized:
    t: np.ndarray        # unix seconds, one per pixel column with data
    v: np.ndarray        # median value (µrad) of trace pixels in the column
    vmin: np.ndarray
    vmax: np.ndarray
    start: int
    end: int
    urad_per_px: float
    y_fit_resid: float   # fraction of OCR'd tick labels that disagree with the fitted axis


def _frame(img: np.ndarray) -> tuple[int, int, int, int]:
    dark = (img.max(axis=2) < 80)
    h, w = dark.shape
    row_frac = dark.sum(axis=1) / w
    col_frac = dark.sum(axis=0) / h
    rows = np.where(row_frac > 0.5)[0]
    cols = np.where(col_frac > 0.5)[0]
    if len(rows) < 2 or len(cols) < 2:
        raise ValueError("axes frame not found")
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def _tick_rows(img: np.ndarray, top: int, bottom: int, left: int) -> list[int]:
    """Y tick marks: short dark horizontal segments just inside the left frame line."""
    dark = img.max(axis=2) < 100
    rows = []
    for r in range(top, bottom + 1):
        seg = dark[r, left + 1:left + 5]
        if seg.all():
            rows.append(r)
    # collapse adjacent rows
    out: list[int] = []
    for r in rows:
        if out and r - out[-1] <= 1:
            continue
        out.append(r)
    return out


_num = re.compile(r"^[-−–]?\d+(\.\d+)?$")


def _ocr_ylabels(img: np.ndarray, top: int, bottom: int, left: int) -> list[tuple[float, float]]:
    """Return [(row_center, value)] from OCR of the y tick labels."""
    if pytesseract is None:
        raise RuntimeError("pytesseract unavailable")
    y0, y1 = max(0, top - 8), min(img.shape[0], bottom + 8)
    x0, x1 = max(0, left - 45), left - 3
    crop = Image.fromarray(img[y0:y1, x0:x1]).convert("L")
    scale = 4
    crop = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
    data = pytesseract.image_to_data(
        crop, config="--psm 6 -c tessedit_char_whitelist=-0123456789.", output_type=pytesseract.Output.DICT
    )
    out = []
    for txt, tp, hgt in zip(data["text"], data["top"], data["height"]):
        s = txt.strip().replace("−", "-").replace("–", "-")
        if s and _num.match(s):
            out.append((y0 + (tp + hgt / 2) / scale, float(s)))
    return out


def _fit_y(ticks: list[int], labels: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Assign labels to tick rows and fit value = a*row + b robustly. Returns (a, b, resid)."""
    if len(ticks) < 3:
        raise ValueError("too few y ticks")
    pairs = []
    for rc, val in labels:
        j = int(np.argmin([abs(rc - t) for t in ticks]))
        if abs(ticks[j] - rc) <= 6:
            pairs.append((ticks[j], val))
    if len(pairs) < 2:
        raise ValueError("too few y labels")
    # consensus: choose the (a, b) from pairs agreeing with the most labels
    best = None
    spacing = np.median(np.diff(ticks))
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            (r1, v1), (r2, v2) = pairs[i], pairs[j]
            if r1 == r2 or v1 == v2:
                continue
            a = (v2 - v1) / (r2 - r1)
            if a >= 0:  # values must decrease downward
                continue
            b = v1 - a * r1
            step = abs(a * spacing)
            inl = [(r, v) for r, v in pairs if abs(a * r + b - v) < 0.2 * step]
            score = (len(inl), -abs(a))
            if best is None or score > best[0]:
                best = (score, inl)
    if best is None or best[0][0] < 3:
        raise ValueError(f"y-axis OCR inconsistent: {pairs}")
    inl = best[1]
    r = np.array([p[0] for p in inl], float)
    v = np.array([p[1] for p in inl], float)
    a, b = np.polyfit(r, v, 1)
    step = abs(a * spacing)
    resid = float(np.max(np.abs(a * r + b - v)) / step)
    return float(a), float(b), resid


NICE_STEPS = [0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100]


def _tick_grid(img: np.ndarray, top: int, bottom: int, left: int) -> list[float]:
    """Full, evenly spaced y-tick grid. The frame's top and bottom are ticks; tick marks that
    are hidden under the trace are recovered from the spacing of the visible ones."""
    seen = _tick_rows(img, top, bottom, left)
    inner = [r for r in seen if top + 2 < r < bottom - 2]
    for n in range(2, 16):
        d = (bottom - top) / n
        grid = [top + k * d for k in range(n + 1)]
        if all(min(abs(r - g) for g in grid) <= 1.5 for r in inner) and (inner or n <= 6):
            return grid
    raise ValueError("irregular y ticks")


def _ocr_label(img: np.ndarray, row: float, left: int) -> float | None:
    y0, y1 = int(round(row)) - 9, int(round(row)) + 9
    x0, x1 = max(0, left - 45), left - 3
    crop = Image.fromarray(img[max(0, y0):y1, x0:x1]).convert("L")
    crop = crop.resize((crop.width * 5, crop.height * 5), Image.LANCZOS)
    crop = crop.point(lambda p: 0 if p < 150 else 255)
    txt = pytesseract.image_to_string(crop, config="--psm 7 -c tessedit_char_whitelist=-0123456789.").strip()
    txt = txt.replace("−", "-").replace("–", "-")
    return float(txt) if _num.match(txt) else None


def _fit_y_grid(grid: list[float], labels: dict[int, float]) -> tuple[float, float, float]:
    """Vote over (nice step, top value): each OCR'd label k implies top = val + k*step."""
    votes: dict[tuple[float, float], int] = {}
    for k, val in labels.items():
        for step in NICE_STEPS:
            key = (step, round(val + k * step, 6))
            votes[key] = votes.get(key, 0) + 1
    if not votes:
        raise ValueError("no y labels read")
    (step, top_val), n = max(votes.items(), key=lambda kv: (kv[1], -abs(kv[0][0])))
    if n < 2:
        raise ValueError(f"y-axis OCR inconsistent: {labels}")
    d = grid[1] - grid[0]
    a = -step / d
    b = top_val - a * grid[0]
    resid = 1.0 - n / max(1, len(labels))  # fraction of labels that disagreed
    return a, b, resid


_range = re.compile(r"(\d{4})-(\d{2})-(\d{2})\s*(\d{2}):(\d{2}):(\d{2})")


def _ocr_xrange(img: np.ndarray, bottom: int) -> tuple[int, int] | None:
    if pytesseract is None:
        return None
    crop = Image.fromarray(img[bottom + 18:min(img.shape[0], bottom + 40), :]).convert("L")
    crop = crop.resize((crop.width * 3, crop.height * 3), Image.LANCZOS)
    txt = pytesseract.image_to_string(crop, config="--psm 7")
    m = _range.findall(txt.replace(" :", ":").replace(": ", ":"))
    if len(m) < 2:
        return None
    ts = []
    for g in m[:2]:
        y, mo, d, hh, mi, ss = map(int, g)
        try:
            ts.append(int(datetime(y, mo, d, hh, mi, ss, tzinfo=HST).timestamp()))
        except ValueError:
            return None
    return ts[0], ts[1]


def trace_mask(img: np.ndarray, color: str = "blue") -> np.ndarray:
    r, g, b = (img[..., i].astype(int) for i in range(3))
    if color == "blue":
        return (b > 150) & (r < 110) & (g < 110)
    return (g > 150) & (r < 110) & (b < 110)


def digitize(png: bytes, expect_span_s: float | None = None,
             fallback_range: tuple[int, int] | None = None, color: str = "blue") -> Digitized:
    img = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
    top, bottom, left, right = _frame(img)
    grid = _tick_grid(img, top, bottom, left)
    labels = {k: v for k, row in enumerate(grid) if (v := _ocr_label(img, row, left)) is not None}
    a, b, resid = _fit_y_grid(grid, labels)

    rng = _ocr_xrange(img, bottom)
    if rng is not None and expect_span_s:
        span = rng[1] - rng[0]
        if not (0.8 * expect_span_s <= span <= 1.25 * expect_span_s):
            rng = None
    if rng is None and fallback_range is not None:
        rng = fallback_range
    if rng is None:
        raise ValueError("could not determine x range")
    start, end = rng

    mask = trace_mask(img, color)
    # ignore legend box area (top-left inside frame): find it by its black border
    legend = _legend_box(img, top, left)
    if legend:
        ly0, ly1, lx0, lx1 = legend
        mask[ly0:ly1 + 1, lx0:lx1 + 1] = False
    mask[: top + 1] = False
    mask[bottom:] = False
    mask[:, : left + 1] = False
    mask[:, right:] = False

    ts, vs, vmins, vmaxs = [], [], [], []
    width = right - left
    for x in range(left + 1, right):
        rows = np.where(mask[:, x])[0]
        if len(rows) == 0:
            continue
        frac = (x - left) / width
        ts.append(start + frac * (end - start))
        vals = a * rows + b
        vs.append(float(np.median(vals)))
        vmins.append(float(vals.min()))
        vmaxs.append(float(vals.max()))
    if len(ts) < 20:
        raise ValueError("trace not found")
    return Digitized(
        t=np.array(ts), v=np.array(vs), vmin=np.array(vmins), vmax=np.array(vmaxs),
        start=start, end=end, urad_per_px=abs(a), y_fit_resid=resid,
    )


def _legend_box(img: np.ndarray, top: int, left: int) -> tuple[int, int, int, int] | None:
    """Legend: a black-bordered box in the top-left of the axes (its sample lines are trace-coloured)."""
    dark = img.max(axis=2) < 130
    for r in range(top + 2, top + 30):
        row = dark[r, left + 2:left + 400]
        cols = np.where(row)[0]
        if len(cols) < 60:
            continue
        # first contiguous dark run = top border of the legend
        x0 = int(cols[0])
        x1 = x0
        while x1 + 1 < len(row) and row[x1 + 1]:
            x1 += 1
        if x1 - x0 < 60:
            continue
        x0, x1 = left + 2 + x0, left + 2 + x1
        for r2 in range(r + 10, min(r + 90, dark.shape[0])):
            if dark[r2, x0:x1 + 1].mean() > 0.9:
                return r, r2, x0, x1
    return None
