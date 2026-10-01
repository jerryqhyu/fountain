"""Seismic tremor (RSAM) from EarthScope FDSN waveforms via ObsPy, one table per sensor.

RSAM = mean absolute amplitude of the 1–5 Hz band-passed vertical velocity (µm/s) in
10-minute windows aligned to the epoch, keyed by window start; a window needs ≥80% data.

UWE and UWE.QC are polled live. Substitute stations (OBL, UWB, RIMD) are only computed
around gaps in UWE∪UWE.QC, with 24 h of context either side for scaling at stitch time.
"""
from __future__ import annotations

import logging
import time

import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException

from .. import db
from ..config import FDSN_BASE, RSAM_BAND, RSAM_WINDOW_S, TREMOR_CHA, TREMOR_NET
from . import SUBSTITUTE_TREMOR, TREMOR_SOURCES, TREMOR_STATIONS
from .base import mark, tracked

log = logging.getLogger("rsam")
TABLE = {s.key: s.table for s in TREMOR_SOURCES}
CHUNK_S = 6 * 3600
_client: Client | None = None


def fdsn() -> Client:
    global _client
    if _client is None:
        _client = Client(FDSN_BASE, timeout=120)
    return _client


def sensitivity(key: str) -> float:
    """Overall sensitivity in counts/(m/s), cached in kv."""
    sta, loc = TREMOR_STATIONS[key]
    k = f"sens:{TREMOR_NET}.{sta}.{loc}.{TREMOR_CHA}"
    if val := db.kv_get(k):
        return float(val)
    inv = fdsn().get_stations(network=TREMOR_NET, station=sta, location=loc or "--", channel=TREMOR_CHA,
                              level="response", endafter=UTCDateTime())
    s = float(inv[0][0][0].response.instrument_sensitivity.value)
    db.kv_set(k, s)
    return s


def compute(key: str, t0: int, t1: int) -> list[tuple[int, float]]:
    """[(window_start, µm/s)] for complete windows in [t0, t1)."""
    sta, loc = TREMOR_STATIONS[key]
    pad = 60
    try:
        st = fdsn().get_waveforms(TREMOR_NET, sta, loc or "--", TREMOR_CHA,
                                  UTCDateTime(t0 - pad), UTCDateTime(t1 + pad))
    except FDSNNoDataException:
        return []
    st = st.select(location=loc)
    if not st:
        return []
    st.merge(method=1, fill_value=None)
    tr = st[0]
    sr = tr.stats.sampling_rate
    data = tr.data
    masked = np.ma.isMaskedArray(data)
    mask = np.ma.getmaskarray(data) if masked else np.zeros(len(data), bool)
    x = np.ma.filled(data.astype(float), np.nan) if masked else data.astype(float)
    fill = np.nanmean(x) if np.isfinite(np.nanmean(x)) else 0.0
    tr2 = tr.copy()
    tr2.data = np.where(np.isnan(x), fill, x)
    tr2.detrend("demean")
    tr2.taper(0.01)
    tr2.filter("bandpass", freqmin=RSAM_BAND[0], freqmax=RSAM_BAND[1], corners=4, zerophase=True)
    y = np.abs(tr2.data)
    y[mask | np.isnan(x)] = np.nan
    start = tr.stats.starttime.timestamp
    sens = sensitivity(key)
    n_win = int(RSAM_WINDOW_S * sr)
    out = []
    w = -(-t0 // RSAM_WINDOW_S) * RSAM_WINDOW_S
    while w + RSAM_WINDOW_S <= t1:
        i0 = int(round((w - start) * sr))
        if i0 >= 0 and i0 + n_win <= len(y):
            seg = y[i0:i0 + n_win]
            good = np.isfinite(seg)
            if good.mean() >= 0.8:
                out.append((int(w), float(np.mean(seg[good])) / sens * 1e6))
        w += RSAM_WINDOW_S
    return out


def compute_range(key: str, t0: int, t1: int, sleep_s: float = 0.3) -> int:
    """Chunked compute + store; skips chunks that are already ≥90% present."""
    table = TABLE[key]
    n = 0
    t = (t0 // CHUNK_S) * CHUNK_S
    while t < t1:
        te = min(t + CHUNK_S, t1)
        have = db.query_one(f"SELECT COUNT(*) AS n FROM {table} WHERE t>=? AND t<?", (t, te))["n"]
        if have < 0.9 * (te - t) / RSAM_WINDOW_S:
            for attempt in range(3):
                try:
                    n += db.upsert(table, ["t", "v"], compute(key, t, te))
                    break
                except Exception as e:  # noqa: BLE001
                    log.warning("rsam %s %s failed (%s)", key, UTCDateTime(t), e)
                    time.sleep(5 * (attempt + 1))
            time.sleep(sleep_s)
        t = te
    return n


def _live(key: str) -> str:
    table = TABLE[key]
    now = int(time.time())
    t1 = (now // RSAM_WINDOW_S) * RSAM_WINDOW_S
    t0 = t1 - 3 * 3600
    last = db.query_one(f"SELECT MAX(t) AS t FROM {table}")["t"]
    if last and t0 - 2 * 86400 < last < t0:
        t0 = int(last)  # fill a short outage
    rows = compute(key, t0, t1)
    db.upsert(table, ["t", "v"], rows)
    if not rows:
        raise RuntimeError("no waveform data in the last 3 h")
    return f"{len(rows)} windows, latest {rows[-1][0]}"


@tracked("uwe")
def fetch_uwe() -> str:
    return _live("uwe")


@tracked("uwe_qc")
def fetch_uwe_qc() -> str:
    return _live("uwe_qc")


def primary_gaps(t0: int, t1: int) -> list[tuple[int, int]]:
    """Window-start ranges in [t0, t1) with neither UWE nor UWE.QC."""
    have = {r["t"] for r in db.query("SELECT t FROM src_rsam_uwe WHERE t>=? AND t<? UNION "
                                     "SELECT t FROM src_rsam_uwe_qc WHERE t>=? AND t<?", (t0, t1, t0, t1))}
    slots = np.arange(-(-t0 // RSAM_WINDOW_S) * RSAM_WINDOW_S, t1, RSAM_WINDOW_S)
    missing = [int(s) for s in slots if int(s) not in have]
    runs: list[tuple[int, int]] = []
    for s in missing:
        if runs and s - runs[-1][1] == RSAM_WINDOW_S:
            runs[-1] = (runs[-1][0], s)
        else:
            runs.append((s, s))
    return runs


def fill_substitutes(t0: int, t1: int, ctx_s: int = 86400) -> str:
    """Compute substitute-station RSAM around each primary gap (first station with data wins)."""
    done, unfilled = 0, 0
    attempted = set(db.kv_get("subst_attempted", []))
    now = time.time()
    for g0, g1 in primary_gaps(t0, t1):
        tag = f"{g0}-{g1}"
        if tag in attempted:
            continue
        if g1 > now - 2 * RSAM_WINDOW_S - 3600:
            continue  # still at the live edge; UWE may yet arrive
        attempted.add(tag)
        for key in SUBSTITUTE_TREMOR:
            try:
                compute_range(key, g0 - ctx_s, g1 + RSAM_WINDOW_S + ctx_s, sleep_s=0.1)
                mark(key, True, detail=f"gap {g0}–{g1}")
            except Exception as e:  # noqa: BLE001
                mark(key, False, error=str(e))
                continue
            n = db.query_one(f"SELECT COUNT(*) AS n FROM {TABLE[key]} WHERE t>=? AND t<=?", (g0, g1))["n"]
            if n:
                done += 1
                break
        else:
            unfilled += 1
    db.kv_set("subst_attempted", sorted(attempted))
    return f"{done} gaps covered by substitutes, {unfilled} with no substitute data"
