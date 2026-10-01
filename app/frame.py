"""The feature frame: one row per 5-minute grid slot, computed from the stitched series.

Every feature at slot t uses only data at or before t, so the same frame serves training
(historical rows) and inference (latest rows). A feature is NaN when its inputs are
genuinely missing; the models then return None for that slot.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import db
from .config import GRID_S
from .episodes import catalog, suggest
from .sources.base import tracked
from .stitch import grid, on_grid

FEATURES = [
    "hours_since_end", "recovery_ratio", "inflation_urad", "gap_to_onset_urad", "last_deflation_urad",
    "tilt_rate_6h", "tilt_rate_24h", "rsam_log", "rsam_ratio_log", "rsam_trend_log",
    "eq_summit_24h", "eq_all_24h", "precursor",
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
SLOTS_PER_H = 3600 // GRID_S


def series(table: str, g: np.ndarray) -> pd.Series:
    rows = db.query(f"SELECT t, v FROM {table} ORDER BY t")
    t = np.array([r["t"] for r in rows], np.int64)
    v = np.array([r["v"] for r in rows], float)
    return pd.Series(on_grid(g, t, v), index=g)


def rate(s: pd.Series, window_h: float) -> pd.Series:
    """Trailing least-squares slope (µrad/h) over the last window_h hours; needs half the window present.

    Computed with window-local x positions (0..n-1) via convolutions, so precision doesn't depend
    on absolute time (rolling sums of epoch-scale x² cancel catastrophically).
    """
    n = int(window_h * SLOTS_PER_H)
    y = s.to_numpy(float)
    m = (~np.isnan(y)).astype(float)
    yz = np.where(m > 0, y, 0.0)
    j = np.arange(n, dtype=float)  # position within the window, oldest = 0

    def wsum(a: np.ndarray, k: np.ndarray) -> np.ndarray:
        # out[i] = sum_{j=0}^{n-1} a[i-n+1+j] * k[j]  (window ending at i)
        return np.convolve(a, k[::-1])[:len(a)]

    S0, S1, S2 = wsum(m, np.ones(n)), wsum(m, j), wsum(m, j * j)
    Sy, Sxy = wsum(yz, np.ones(n)), wsum(yz, j)
    den = S0 * S2 - S1 * S1
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = (S0 * Sxy - S1 * Sy) / den  # per slot
    slope[(S0 < max(3, n // 2)) | (den <= 0)] = np.nan
    slope[: n - 1] = np.nan  # incomplete leading windows
    return pd.Series(slope * SLOTS_PER_H, index=s.index)


def episode_table(tilt: pd.Series) -> pd.DataFrame:
    """Fountaining episodes with tilt at onset, trough after the episode, and deflation."""
    rows = []
    for e in catalog.fountaining():
        pre = tilt[(tilt.index > e["start_t"] - 3600) & (tilt.index <= e["start_t"])].dropna()
        onset_v = float(pre.median()) if len(pre) else None
        tr = None
        if e["end_t"]:
            w = tilt[(tilt.index >= e["start_t"]) & (tilt.index <= e["end_t"] + 2 * 3600)].dropna()
            tr = float(w.min()) if len(w) else None
        rows.append({"label": e["label"], "num": e["num"], "start": e["start_t"], "end": e["end_t"],
                     "onset_v": onset_v, "trough_v": tr,
                     "defl": (onset_v - tr) if onset_v is not None and tr is not None else None})
    return pd.DataFrame(rows)


def annotate_episodes(eps: pd.DataFrame) -> None:
    """Store deflation and onset-recovery metrics on the catalog (shown in the episode table)."""
    ups, prev = [], None
    for _, e in eps.iterrows():
        rec = ratio = None
        if prev is not None and pd.notna(prev["trough_v"]) and pd.notna(e["onset_v"]):
            rec = e["onset_v"] - prev["trough_v"]
            if pd.notna(prev["defl"]) and prev["defl"] > 0.5:
                ratio = rec / prev["defl"]
        ups.append((None if pd.isna(e["defl"]) else float(e["defl"]), rec, ratio, e["label"]))
        prev = e
    with db.tx() as c:
        c.executemany("UPDATE episodes SET deflation_urad=?, onset_recovery_urad=?, onset_recovery_ratio=? "
                      "WHERE label=?", ups)


def build(t0: int | None = None, t1: int | None = None) -> pd.DataFrame:
    """Frame rows for slots in [t0, t1] (defaults: whole grid)."""
    g = grid(t1)
    tilt = series("series_tilt", g)
    rsam = series("series_rsam", g)
    eps = episode_table(tilt)
    annotate_episodes(eps)

    df = pd.DataFrame(index=g)
    df.index.name = "t"
    starts = eps["start"].to_numpy(np.int64)
    ends = eps["end"].fillna(2**62).to_numpy(np.int64)
    last = np.searchsorted(ends, g, side="right") - 1
    nxt = np.searchsorted(starts, g, side="right")
    has = last >= 0
    li = np.where(has, last, 0)

    in_ep = np.zeros(len(g), bool)
    for ev in db.query("SELECT start_t, end_t FROM episodes"):
        in_ep |= (g >= ev["start_t"]) & (g <= (ev["end_t"] or 2**62))
    df["in_episode"] = in_ep.astype(float)
    df["last_label"] = np.where(has, eps["label"].to_numpy()[li], None)
    df["last_end"] = np.where(has, ends[li], np.nan).astype(float)
    df["next_onset"] = np.where(nxt < len(starts), starts[np.minimum(nxt, len(starts) - 1)], np.nan).astype(float)

    v = tilt.rolling(SLOTS_PER_H, min_periods=SLOTS_PER_H // 2).median().to_numpy()
    trough = eps["trough_v"].to_numpy(float)[li]
    onset = eps["onset_v"].to_numpy(float)[li]
    defl = eps["defl"].to_numpy(float)[li]
    df["hours_since_end"] = np.where(has, (g - df["last_end"].to_numpy()) / 3600.0, np.nan)
    df["inflation_urad"] = np.where(has, v - trough, np.nan)
    df["gap_to_onset_urad"] = np.where(has, v - onset, np.nan)
    df["last_deflation_urad"] = np.where(has, defl, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["recovery_ratio"] = np.where(has & (defl > 0.5), (v - trough) / defl, np.nan)
    df["tilt_rate_6h"] = rate(tilt, 6).to_numpy()
    df["tilt_rate_24h"] = rate(tilt, 24).to_numpy()
    df["tilt_value"] = v

    r1 = rsam.rolling(SLOTS_PER_H, min_periods=SLOTS_PER_H // 2).median()
    r6 = rsam.rolling(6 * SLOTS_PER_H, min_periods=3 * SLOTS_PER_H).median()
    r24 = rsam.rolling(24 * SLOTS_PER_H, min_periods=12 * SLOTS_PER_H).median()
    with np.errstate(invalid="ignore", divide="ignore"):
        df["rsam_log"] = np.log10(r1).to_numpy()
        df["rsam_ratio_log"] = np.log10(r1 / r24).to_numpy()
        df["rsam_trend_log"] = np.log10(r1 / r6).to_numpy()
    df["rsam_1h_ums"] = r1.to_numpy()

    def counts(region: str | None) -> np.ndarray:
        sql = "SELECT t FROM earthquakes" + (" WHERE region=?" if region else "") + " ORDER BY t"
        ev = np.array([r["t"] for r in db.query(sql, (region,) if region else ())], np.int64)
        return (np.searchsorted(ev, g, side="right") - np.searchsorted(ev, g - 86400, side="right")).astype(float)

    df["eq_summit_24h"] = counts("summit")
    df["eq_all_24h"] = counts(None)

    ps = suggest.precursor_series()
    prec = np.zeros(len(g))
    if ps:
        pt = np.array([t for t, _ in ps], np.int64)
        pf = np.array([f for _, f in ps], float)
        pos = np.searchsorted(pt, g, side="right") - 1
        ok = (pos >= 0)
        ok[ok] &= (g[ok] - pt[pos[ok]]) <= 36 * 3600
        prec[ok] = pf[pos[ok]]
    df["precursor"] = prec

    lo = t0 if t0 is not None else g[0]
    return df[df.index >= lo]


def write(df: pd.DataFrame) -> int:
    cols = db.FRAME_COLUMNS
    out = df[cols].astype(object).where(df[cols].notna(), None)
    return db.upsert("frame", ["t"] + cols, ([int(t)] + list(r) for t, r in zip(out.index, out.itertuples(False))))


def load(t0: int | None = None, t1: int | None = None) -> pd.DataFrame:
    sql, p = "SELECT * FROM frame WHERE 1=1", []
    if t0 is not None:
        sql += " AND t>=?"; p.append(t0)
    if t1 is not None:
        sql += " AND t<=?"; p.append(t1)
    df = pd.DataFrame(db.query(sql + " ORDER BY t", p))
    return df.set_index("t") if len(df) else pd.DataFrame(columns=["t"] + db.FRAME_COLUMNS).set_index("t")


@tracked("frame")
def run(t0: int | None = None) -> str:
    t = time.time()
    df = build(t0)
    n = write(df)
    return f"{n} slots from {int(df.index[0])} ({time.time() - t:.1f} s)"
