"""Live UWD tilt, digitized from HVO's plot PNGs (see app/digitize.py).

USGS publishes live tilt only as plot images (the CSV release lags ~60 days; UWD is not
on EarthScope FDSN). Two plots are used: the 2-day plot (live, fine) and the 3-month plot
(bridges the release lag on a fresh install).

Each digitization is interpolated onto the 5-minute grid and stored in the plot's own
offset (values are aligned to the authoritative series at stitch time). A slot keeps its
first recorded value; only the last TAIL_REFRESH_S of the table may be revised, so the
hourly re-digitizing of a coarse plot cannot make history wobble.
"""
from __future__ import annotations

import time
from email.utils import parsedate_to_datetime

import numpy as np

from .. import db, digitize
from ..config import GRID_S
from .base import get, tracked

BASE = "https://volcanoes.usgs.gov/vsc/captures/kilauea/"
PLOTS = {
    # key: (file, span seconds, table)
    "uwd_plot2d": ("UWD-TILT-2day.png", 2 * 86400, "src_tilt_uwd_plot2d"),
    "uwd_plot3m": ("UWD-TILT-3month.png", 90 * 86400, "src_tilt_uwd_plot3m"),
}
TAIL_REFRESH_S = 6 * 3600


def to_grid(d: digitize.Digitized) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate pixel-column samples onto grid slots; no bridging across >4 px holes."""
    px = (d.end - d.start) / 745
    g = np.arange(int(np.ceil(d.t[0] / GRID_S)) * GRID_S, int(d.t[-1]) + 1, GRID_S)
    v = np.interp(g, d.t, d.v)
    i = np.clip(np.searchsorted(d.t, g), 1, len(d.t) - 1)
    ok = (d.t[i] - d.t[i - 1]) <= max(4 * px, 2 * GRID_S)
    return g[ok], v[ok]


def store(table: str, slots: np.ndarray, vals: np.ndarray) -> int:
    last = db.query_one(f"SELECT MAX(t) AS t FROM {table}")["t"] or 0
    have = {r["t"] for r in db.query(f"SELECT t FROM {table} WHERE t >= ?", (int(slots[0]),))}
    now = int(time.time())
    rows = [(int(t), float(v), now) for t, v in zip(slots, vals)
            if int(t) not in have or int(t) > last - TAIL_REFRESH_S]
    return db.upsert(table, ["t", "v", "fetched_at"], rows)


def _fetch(key: str) -> str:
    file, span, table = PLOTS[key]
    r = get(BASE + file)
    r.raise_for_status()
    if not r.headers.get("content-type", "").startswith("image/"):
        raise ValueError(f"not an image ({r.headers.get('content-type')})")
    fallback = None
    if lm := r.headers.get("last-modified"):
        end = int(parsedate_to_datetime(lm).timestamp())
        fallback = (end - span, end)
    d = digitize.digitize(r.content, expect_span_s=span, fallback_range=fallback)
    if d.y_fit_resid > 0.6:
        raise ValueError(f"y-axis labels mostly unreadable ({d.y_fit_resid:.0%} disagree)")
    slots, vals = to_grid(d)
    n = store(table, slots, vals)
    db.kv_set(f"plot_meta:{key}", {"start": d.start, "end": d.end, "urad_per_px": d.urad_per_px,
                                   "label_disagreement": d.y_fit_resid, "fetched_at": int(time.time())})
    return f"{len(slots)} slots digitized, {n} written; plot ends {d.end}"


@tracked("uwd_plot2d")
def fetch_2day() -> str:
    return _fetch("uwd_plot2d")


@tracked("uwd_plot3m")
def fetch_3month() -> str:
    return _fetch("uwd_plot3m")
