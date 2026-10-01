"""Live UWD tilt, digitized from the HVO plot PNGs.

Tilt is not available through EarthScope FDSN (checked: no HV.UWD channels in the station
service), and the USGS CSV release lags ~60 days, so the live tail comes from the
official plots. See analytics/digitize.py for how, and analytics/tilt.py for stitching
the digitized plots onto the CSV release baseline.
"""
from __future__ import annotations

import logging
import time
from email.utils import parsedate_to_datetime

from .. import db
from ..analytics import digitize
from .base import get, tracked

log = logging.getLogger("usgs_tilt")

BASE = "https://volcanoes.usgs.gov/vsc/captures/kilauea/"
PLOTS = {
    # name: (file, span seconds)
    "2day": ("UWD-TILT-2day.png", 2 * 86400),
    "week": ("UWD-TILT-week.png", 7 * 86400),
    "month": ("UWD-TILT-month.png", 30 * 86400),
    "3month": ("UWD-TILT-3month.png", 90 * 86400),
}
# Wayback Machine copy of the 3-month plot captured 2026-08-06. It spans 2026-05-08 ..
# 2026-08-06 and bridges the end of the CSV release (May 31) to the live 3-month plot.
ARCHIVED = {
    "wb_3month_20260806": (
        "https://web.archive.org/web/20260806084150im_/https://volcanoes.usgs.gov/vsc/captures/kilauea/UWD-TILT-3month.png",
        90 * 86400,
    ),
}


def _store(name: str, d: digitize.Digitized) -> None:
    now = int(time.time())
    with db.tx() as c:
        c.execute("DELETE FROM tilt_plot WHERE plot=?", (name,))
        c.executemany(
            "INSERT OR REPLACE INTO tilt_plot(plot,t,v,vmin,vmax,fetched_at) VALUES(?,?,?,?,?,?)",
            [(name, int(t), float(v), float(a), float(b), now) for t, v, a, b in zip(d.t, d.v, d.vmin, d.vmax)],
        )
    db.kv_set(f"tilt_plot_meta:{name}", {
        "start": d.start, "end": d.end, "urad_per_px": d.urad_per_px, "resid": d.y_fit_resid, "fetched_at": now,
    })


def fetch_plot(name: str, url: str, span: int) -> digitize.Digitized:
    r = get(url)
    r.raise_for_status()
    if not r.headers.get("content-type", "").startswith("image/"):
        raise ValueError(f"{name}: not an image ({r.headers.get('content-type')})")
    fallback = None
    lm = r.headers.get("last-modified")
    if lm:
        end = int(parsedate_to_datetime(lm).timestamp())
        fallback = (end - span, end)
    d = digitize.digitize(r.content, expect_span_s=span, fallback_range=fallback)
    if d.y_fit_resid > 0.6:
        raise ValueError(f"{name}: y-axis labels mostly unreadable ({d.y_fit_resid:.0%} disagree)")
    _store(name, d)
    return d


def _run(names: list[str]) -> str:
    out, errors = [], []
    for name in names:
        file, span = PLOTS[name]
        try:
            d = fetch_plot(name, BASE + file, span)
            out.append(f"{name}:{len(d.t)}px")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e}")
            log.warning("tilt plot %s failed: %s", name, e)
    for name, (url, span) in ARCHIVED.items():
        if not db.query_one("SELECT 1 FROM tilt_plot WHERE plot=? LIMIT 1", (name,)):
            try:
                d = fetch_plot(name, url, span)
                out.append(f"{name}:{len(d.t)}px")
            except Exception as e:  # noqa: BLE001
                log.warning("archived tilt plot %s failed: %s", name, e)
    from ..analytics import tilt as tilt_an
    tilt_an.rebuild_canonical()
    if not out:
        raise RuntimeError("; ".join(errors))
    return ", ".join(out) + (f" (errors: {'; '.join(errors)})" if errors else "")


@tracked("usgs_tilt")
def fetch_fast() -> str:
    """Every 10 minutes: the high-resolution plots."""
    return _run(["2day", "week"])


@tracked("usgs_tilt_long")
def fetch_slow() -> str:
    """Hourly: the long-range plots (they only refresh hourly upstream)."""
    return _run(["month", "3month"])
