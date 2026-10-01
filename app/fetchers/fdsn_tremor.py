"""Seismic tremor (RSAM) from EarthScope FDSN waveforms via ObsPy.

RSAM here = mean absolute amplitude of the 1–5 Hz band-passed vertical velocity in
10-minute windows aligned to the epoch. Stored both in raw counts and in µm/s.
"""
from __future__ import annotations

import logging
import time

import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException

from .. import db
from ..config import (
    FDSN_BASE,
    RSAM_BAND,
    RSAM_WINDOW_S,
    TREMOR_CHA,
    TREMOR_LOC,
    TREMOR_NET,
    TREMOR_STA,
)
from .base import tracked

log = logging.getLogger("tremor")
_client: Client | None = None


def fdsn() -> Client:
    global _client
    if _client is None:
        _client = Client(FDSN_BASE, timeout=120)
    return _client


# The co-located UWE "QC" sensor fills outages of the primary (e.g. 2026-08-15..09-15).
# In µm/s the two agree to ~1% (checked on 2026-09-26).
FALLBACK_LOCS = ["QC"]


def sensitivity(loc: str = TREMOR_LOC) -> float:
    """Overall sensitivity in counts/(m/s); cached in kv."""
    key = f"sens:{TREMOR_NET}.{TREMOR_STA}.{loc}.{TREMOR_CHA}"
    val = db.kv_get(key)
    if val:
        return float(val)
    inv = fdsn().get_stations(
        network=TREMOR_NET, station=TREMOR_STA, location=loc or "--",
        channel=TREMOR_CHA, level="response", endafter=UTCDateTime(),
    )
    s = float(inv[0][0][0].response.instrument_sensitivity.value)
    db.kv_set(key, s)
    return s


def compute_rsam(t0: int, t1: int) -> list[tuple[int, float, float]]:
    """RSAM windows in [t0, t1) from the primary sensor, gaps filled from the fallback sensor."""
    rows = _compute_loc(t0, t1, TREMOR_LOC)
    expected = (t1 - t0) // RSAM_WINDOW_S
    if len(rows) < expected:
        have = {r[0] for r in rows}
        for loc in FALLBACK_LOCS:
            extra = [r for r in _compute_loc(t0, t1, loc) if r[0] not in have]
            rows += extra
            have |= {r[0] for r in extra}
            if len(rows) >= expected:
                break
        rows.sort()
    return rows


def _compute_loc(t0: int, t1: int, loc: str) -> list[tuple[int, float, float]]:
    """Return [(window_start_unix, counts, um_per_s)] for complete windows in [t0, t1)."""
    pad = 60
    try:
        st = fdsn().get_waveforms(
            TREMOR_NET, TREMOR_STA, loc or "--", TREMOR_CHA,
            UTCDateTime(t0 - pad), UTCDateTime(t1 + pad),
        )
    except FDSNNoDataException:
        return []
    st = st.select(location=loc)
    if not st:
        return []
    st.merge(method=1, fill_value=None)
    tr = st[0]
    sr = tr.stats.sampling_rate
    data = tr.data
    mask = np.ma.getmaskarray(data) if np.ma.isMaskedArray(data) else np.zeros(len(data), bool)
    x = np.ma.filled(data.astype(float), np.nan) if np.ma.isMaskedArray(data) else data.astype(float)
    # Filter on a gap-free copy (gaps filled with the local mean) then re-mask.
    fill = np.nanmean(x) if np.isfinite(np.nanmean(x)) else 0.0
    xf = np.where(np.isnan(x), fill, x)
    tr2 = tr.copy()
    tr2.data = xf
    tr2.detrend("demean")
    tr2.taper(0.01)
    tr2.filter("bandpass", freqmin=RSAM_BAND[0], freqmax=RSAM_BAND[1], corners=4, zerophase=True)
    y = np.abs(tr2.data)
    y[mask | np.isnan(x)] = np.nan
    start = tr.stats.starttime.timestamp
    sens = sensitivity(loc)
    out = []
    w0 = (t0 // RSAM_WINDOW_S) * RSAM_WINDOW_S
    if w0 < t0:
        w0 += RSAM_WINDOW_S
    n_win = int(RSAM_WINDOW_S * sr)
    w = w0
    while w + RSAM_WINDOW_S <= t1:
        i0 = int(round((w - start) * sr))
        i1 = i0 + n_win
        if i0 >= 0 and i1 <= len(y):
            seg = y[i0:i1]
            good = np.isfinite(seg)
            if good.mean() >= 0.8:
                c = float(np.mean(seg[good]))
                out.append((int(w), c, c / sens * 1e6))
        w += RSAM_WINDOW_S
    return out


def store(rows: list[tuple[int, float, float]]) -> None:
    if not rows:
        return
    with db.tx() as c:
        c.executemany(
            "INSERT OR REPLACE INTO rsam(station,t,counts,ums) VALUES(?,?,?,?)",
            [(TREMOR_STA, t, cnt, u) for t, cnt, u in rows],
        )


@tracked("fdsn_tremor")
def fetch() -> str:
    """Live job: recompute the last ~3 hours (catches late-arriving packets)."""
    now = int(time.time())
    t1 = (now // RSAM_WINDOW_S) * RSAM_WINDOW_S
    t0 = t1 - 3 * 3600
    last = db.query_one("SELECT MAX(t) AS t FROM rsam WHERE station=?", (TREMOR_STA,))
    if last and last["t"] and last["t"] < t0 and t0 - last["t"] < 2 * 86400:
        t0 = int(last["t"])  # fill a short outage
    rows = compute_rsam(t0, t1)
    store(rows)
    if not rows:
        raise RuntimeError("no waveform data returned")
    return f"{len(rows)} windows, latest {rows[-1][0]}"


def backfill(start: int, end: int, chunk_s: int = 6 * 3600, sleep_s: float = 0.5) -> None:
    """Fill history chunk by chunk, skipping chunks already mostly present."""
    t = (start // chunk_s) * chunk_s
    while t < end:
        t_end = min(t + chunk_s, end)
        have = db.query_one(
            "SELECT COUNT(*) AS n FROM rsam WHERE station=? AND t>=? AND t<?", (TREMOR_STA, t, t_end)
        )["n"]
        expected = (t_end - t) // RSAM_WINDOW_S
        if have < 0.9 * expected:
            for attempt in range(3):
                try:
                    rows = compute_rsam(t, t_end)
                    store(rows)
                    log.info("rsam backfill %s: %d windows", UTCDateTime(t).isoformat(), len(rows))
                    break
                except Exception as e:  # noqa: BLE001
                    log.warning("rsam backfill %s failed (%s), retrying", UTCDateTime(t), e)
                    time.sleep(10 * (attempt + 1))
            time.sleep(sleep_s)
        t = t_end
