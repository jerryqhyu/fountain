"""Feature construction shared by training (hourly history) and live prediction.

Every feature at time t uses only data with timestamps <= t (no look-ahead), so the
same code produces training rows and the live row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import db
from ..analytics import quakes as quakes_an
from ..analytics import tilt as tilt_an
from ..analytics import tremor as tremor_an
from ..episodes import catalog, suggest

FEATURES = [
    "hours_since_end",
    "recovery_ratio",
    "inflation_urad",
    "gap_to_onset_urad",
    "last_deflation_urad",
    "tilt_rate_6h",
    "tilt_rate_24h",
    "rsam_log",
    "rsam_ratio_log",
    "rsam_trend_log",
    "eq_summit_24h",
    "eq_all_24h",
    "precursor",
]

LABELS = {
    "hours_since_end": "Time since last episode ended",
    "recovery_ratio": "Tilt recovery vs. last deflation",
    "inflation_urad": "Inflation since last episode",
    "gap_to_onset_urad": "Tilt vs. last onset level",
    "last_deflation_urad": "Size of last deflation",
    "tilt_rate_6h": "Tilt rate (6 h)",
    "tilt_rate_24h": "Tilt rate (24 h)",
    "rsam_log": "Tremor level (RSAM)",
    "rsam_ratio_log": "Tremor 1 h vs 24 h",
    "rsam_trend_log": "Tremor 1 h vs 6 h",
    "eq_summit_24h": "Summit earthquakes (24 h)",
    "eq_all_24h": "Regional earthquakes (24 h)",
    "precursor": "HVO reports precursory activity",
}


def _asof(series: pd.Series, times: np.ndarray, tol_s: int) -> np.ndarray:
    if series.empty:
        return np.full(len(times), np.nan)
    idx = series.index.to_numpy(np.int64)
    vals = series.to_numpy(float)
    pos = np.searchsorted(idx, times, side="right") - 1
    out = np.full(len(times), np.nan)
    ok = pos >= 0
    ok[ok] &= (times[ok] - idx[pos[ok]]) <= tol_s
    out[ok] = vals[pos[ok]]
    return out


def episode_table() -> pd.DataFrame:
    """Fountaining episodes with onset tilt level, trough and deflation."""
    s = tilt_an.load()
    rows = []
    for e in catalog.fountaining():
        onset_v = tilt_an.value_at(s, e["start_t"])
        tr = tilt_an.trough(s, e["start_t"], e["end_t"]) if e["end_t"] else None
        rows.append({
            "label": e["label"], "num": e["num"], "start": e["start_t"], "end": e["end_t"],
            "onset_v": onset_v, "trough_v": tr,
            "defl": (onset_v - tr) if onset_v is not None and tr is not None else None,
        })
    return pd.DataFrame(rows)


def build(times: np.ndarray, eps: pd.DataFrame | None = None) -> pd.DataFrame:
    times = np.asarray(times, dtype=np.int64)
    eps = episode_table() if eps is None else eps
    all_events = db.query("SELECT start_t, end_t FROM episodes ORDER BY start_t")

    df = pd.DataFrame({"t": times})
    starts = eps["start"].to_numpy(np.int64)
    ends = eps["end"].fillna(np.iinfo(np.int64).max).to_numpy(np.int64)

    # last fountaining episode that has *ended* by t, and the next onset after t
    last_idx = np.searchsorted(ends, times, side="right") - 1
    next_idx = np.searchsorted(starts, times, side="right")
    df["cycle"] = last_idx
    in_ep = np.zeros(len(times), bool)
    for ev in all_events:
        e_end = ev["end_t"] or np.iinfo(np.int64).max
        in_ep |= (times >= ev["start_t"]) & (times <= e_end)
    df["in_episode"] = in_ep
    has_last = last_idx >= 0
    li = np.where(has_last, last_idx, 0)
    df["last_label"] = np.where(has_last, eps["label"].to_numpy()[li], None)
    df["last_end"] = np.where(has_last, ends[li], np.nan)
    df["next_onset"] = np.where(next_idx < len(starts), starts[np.minimum(next_idx, len(starts) - 1)], np.nan)

    # tilt
    s = tilt_an.load()
    s_med = s.rolling(6, min_periods=1).median() if not s.empty else s
    v = _asof(s_med, times, 3600)
    r6 = _asof(tilt_an.rate_series(s, 6.0), times, 3600)
    r24 = _asof(tilt_an.rate_series(s, 24.0), times, 3600)
    trough_v = eps["trough_v"].to_numpy(float)[li]
    onset_v = eps["onset_v"].to_numpy(float)[li]
    defl = eps["defl"].to_numpy(float)[li]
    df["hours_since_end"] = np.where(has_last, (times - df["last_end"].to_numpy()) / 3600.0, np.nan)
    df["inflation_urad"] = np.where(has_last, v - trough_v, np.nan)
    df["gap_to_onset_urad"] = np.where(has_last, v - onset_v, np.nan)
    df["last_deflation_urad"] = np.where(has_last, defl, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["recovery_ratio"] = np.where(has_last & (defl > 0.5), (v - trough_v) / defl, np.nan)
    df["tilt_rate_6h"] = r6
    df["tilt_rate_24h"] = r24
    df["tilt_value"] = v

    # tremor
    rs = tremor_an.load()
    rf = tremor_an.rolling_features(rs)
    r1h = _asof(rf["r1h"].dropna(), times, 1800) if len(rf) else np.full(len(times), np.nan)
    r6h = _asof(rf["r6h"].dropna(), times, 1800) if len(rf) else np.full(len(times), np.nan)
    r24h = _asof(rf["r24h"].dropna(), times, 3600) if len(rf) else np.full(len(times), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["rsam_log"] = np.log10(r1h)
        df["rsam_ratio_log"] = np.log10(r1h / r24h)
        df["rsam_trend_log"] = np.log10(r1h / r6h)
    df["rsam_1h_ums"] = r1h

    # earthquakes
    df["eq_summit_24h"] = quakes_an.count_series(times, "summit", 86400)
    df["eq_all_24h"] = quakes_an.count_series(times, None, 86400)

    # precursory activity reported in the latest HVO update within 36 h
    ps = suggest.precursor_series()
    if ps:
        pser = pd.Series([f for _, f in ps], index=np.array([t for t, _ in ps], dtype=np.int64))
        pser = pser[~pser.index.duplicated(keep="last")]
        df["precursor"] = np.nan_to_num(_asof(pser, times, 36 * 3600), nan=0.0)
    else:
        df["precursor"] = 0.0
    return df


def labels(df: pd.DataFrame, horizon_h: float, now: int) -> np.ndarray:
    """1 if the next fountaining onset is within (t, t+H]; 0 if not; NaN if censored."""
    h = horizon_h * 3600
    t = df["t"].to_numpy(float)
    nxt = df["next_onset"].to_numpy(float)
    y = np.full(len(df), np.nan)
    known = ~np.isnan(nxt)
    y[known] = ((nxt[known] - t[known]) <= h).astype(float)
    # no onset has happened yet: negative only if the full horizon has already elapsed
    unk = ~known & (t + h <= now)
    y[unk] = 0.0
    return y
