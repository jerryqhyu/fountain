"""Tilt analytics: stitched UWD az-300 series, tilt rate, inflation since last episode.

Canonical series (`tilt` table):
  * src='release:*'  – USGS 1-min CSV release, 10-min means (authoritative)
  * src='plot:*'     – digitized plots, offset-aligned to whatever precedes them, used only
                       after the end of the release. Finer plots override coarser ones, and
                       recorded history is otherwise frozen (see rebuild_canonical).
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from .. import db

log = logging.getLogger("tilt")
STEP = 600
LAYERS = ["wb_3month_20260806", "3month", "month", "week", "2day"]


def _series(rows: list[dict], col: str = "v") -> pd.Series:
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series([r[col] for r in rows], index=np.array([r["t"] for r in rows], dtype=np.int64)).sort_index()


def _to_grid(s: pd.Series, max_gap_s: int) -> pd.Series:
    """Resample an irregular series onto the 10-min grid by linear interpolation (no long gaps)."""
    if len(s) < 2:
        return pd.Series(dtype=float)
    g0 = int(np.ceil(s.index[0] / STEP) * STEP)
    g1 = int(np.floor(s.index[-1] / STEP) * STEP)
    grid = np.arange(g0, g1 + 1, STEP)
    x = s.index.to_numpy(float)
    v = np.interp(grid, x, s.to_numpy(float))
    # blank grid points that fall inside gaps of the source
    idx = np.searchsorted(x, grid)
    idx = np.clip(idx, 1, len(x) - 1)
    gap = x[idx] - x[idx - 1]
    v[gap > max_gap_s] = np.nan
    return pd.Series(v, index=grid).dropna()


RANK = {name: i for i, name in enumerate(LAYERS)}  # higher = finer resolution
TAIL_REFRESH_S = 6 * 3600


def _rank(src: str) -> int:
    if src.startswith("release:") or src == "release":
        return 99
    return RANK.get(src.split(":", 1)[-1], -1)


def rebuild_canonical() -> dict:
    """Merge newly digitized plots into the canonical series.

    Recorded history is frozen: a time slot already written is only replaced by a strictly
    finer plot, or if it lies in the last few hours (where the live plot may still refine it).
    Re-digitizing a coarse plot every hour shifts its pixel columns slightly, and without this
    rule past values (e.g. an episode's deflation) would wobble from hour to hour.
    """
    rel = _series(db.query("SELECT t, v FROM tilt WHERE src LIKE 'release:%' ORDER BY t"))
    if rel.empty:
        return {"error": "no release data"}
    release_end = int(rel.index[-1])
    old = db.query("SELECT t, v, src FROM tilt WHERE src LIKE 'plot:%' AND t > ? ORDER BY t", (release_end,))
    comp = pd.concat([rel, _series(old)]).sort_index()
    comp_src = pd.concat([pd.Series("release", index=rel.index),
                          pd.Series([r["src"] for r in old], index=np.array([r["t"] for r in old], dtype=np.int64))]).sort_index()
    tail_cut = (int(old[-1]["t"]) if old else release_end) - TAIL_REFRESH_S
    changed: dict[int, tuple[float, str]] = {}
    info = {}
    for name in LAYERS:
        meta = db.kv_get(f"tilt_plot_meta:{name}")
        rows = db.query("SELECT t, v FROM tilt_plot WHERE plot=? ORDER BY t", (name,))
        if not rows or not meta:
            continue
        span = meta["end"] - meta["start"]
        px_s = span / 745
        lay = _to_grid(_series(rows), max_gap_s=int(max(4 * px_s, 1800)))
        if lay.empty:
            continue
        common = comp.index.intersection(lay.index)
        # align on the most recent overlap (up to 10 days)
        common = common[common >= (common.max() - 10 * 86400)] if len(common) else common
        if len(common) < 12:
            info[name] = "no overlap"
            continue
        offset = float(np.median(comp.loc[common].to_numpy() - lay.loc[common].to_numpy()))
        resid = float(np.median(np.abs(comp.loc[common].to_numpy() - lay.loc[common].to_numpy() - offset)))
        shifted = (lay + offset)
        shifted = shifted[shifted.index > release_end]
        rank = RANK[name]
        cur = comp_src.reindex(shifted.index)
        cur_rank = np.array([_rank(x) if isinstance(x, str) else -1 for x in cur], dtype=int)
        t_idx = shifted.index.to_numpy(np.int64)
        take = (cur_rank < 0) | (rank > cur_rank) | ((t_idx > tail_cut) & (rank >= cur_rank) & (cur_rank < 99))
        upd = shifted[take]
        if len(upd):
            comp = pd.concat([comp[~comp.index.isin(upd.index)], upd]).sort_index()
            comp_src = pd.concat([comp_src[~comp_src.index.isin(upd.index)],
                                  pd.Series(f"plot:{name}", index=upd.index)]).sort_index()
            for t, v in upd.items():
                changed[int(t)] = (float(v), f"plot:{name}")
        info[name] = {"offset": round(offset, 3), "mad": round(resid, 3), "n_overlap": int(len(common)),
                      "updated": int(len(upd))}
    if changed:
        with db.tx() as c:
            c.executemany("INSERT OR REPLACE INTO tilt(t,v,src) VALUES(?,?,?)",
                          [(t, v, src) for t, (v, src) in changed.items()])
    db.kv_set("tilt_stitch", {"release_end": release_end, "layers": info, "built_at": int(time.time())})
    return info


def load(t0: int | None = None, t1: int | None = None) -> pd.Series:
    sql, p = "SELECT t, v FROM tilt WHERE 1=1", []
    if t0 is not None:
        sql += " AND t>=?"; p.append(t0)
    if t1 is not None:
        sql += " AND t<=?"; p.append(t1)
    return _series(db.query(sql + " ORDER BY t", p))


def rate_series(s: pd.Series, window_h: float = 3.0) -> pd.Series:
    """Trailing least-squares slope in µrad/hour on the 10-min grid."""
    if len(s) < 4:
        return pd.Series(dtype=float)
    g = s.reindex(np.arange(s.index[0], s.index[-1] + 1, STEP))
    n = int(window_h * 3600 / STEP)
    x = pd.Series(g.index.to_numpy(float) / 3600.0, index=g.index)
    y = g
    mask = y.notna()
    xm = x.where(mask)
    cnt = mask.astype(float).rolling(n, min_periods=max(3, n // 2)).sum()
    sx = xm.rolling(n, min_periods=max(3, n // 2)).sum()
    sy = y.rolling(n, min_periods=max(3, n // 2)).sum()
    sxx = (xm * xm).rolling(n, min_periods=max(3, n // 2)).sum()
    sxy = (xm * y).rolling(n, min_periods=max(3, n // 2)).sum()
    den = cnt * sxx - sx * sx
    slope = (cnt * sxy - sx * sy) / den.replace(0, np.nan)
    return slope.dropna()


def value_at(s: pd.Series, t: int, before_s: int = 3600) -> float | None:
    w = s[(s.index > t - before_s) & (s.index <= t)]
    return float(w.median()) if len(w) else None


def trough(s: pd.Series, start: int, end: int) -> float | None:
    w = s[(s.index >= start) & (s.index <= end + 2 * 3600)]
    return float(w.min()) if len(w) else None


def annotate_episodes() -> None:
    """Store deflation and onset-recovery metrics for each fountaining episode."""
    from ..episodes import catalog

    s = load()
    eps = catalog.fountaining()
    prev = None
    updates = []
    for e in eps:
        onset_v = value_at(s, e["start_t"])
        tr = trough(s, e["start_t"], e["end_t"]) if e["end_t"] else None
        defl = onset_v - tr if onset_v is not None and tr is not None else None
        rec = ratio = None
        if prev and prev.get("_trough") is not None and onset_v is not None:
            rec = onset_v - prev["_trough"]
            if prev.get("_defl"):
                ratio = rec / prev["_defl"] if prev["_defl"] > 0.5 else None
        e["_trough"], e["_defl"] = tr, defl
        updates.append((defl, rec, ratio, e["label"]))
        prev = e
    with db.tx() as c:
        c.executemany(
            "UPDATE episodes SET deflation_urad=?, onset_recovery_urad=?, onset_recovery_ratio=? WHERE label=?",
            updates,
        )


def current_state(now: int | None = None) -> dict:
    """Tilt features at time `now` (defaults to latest data)."""
    from ..episodes import catalog

    s = load()
    if s.empty:
        return {"available": False}
    now = int(now or s.index[-1])
    s_now = s[s.index <= now]
    if s_now.empty:
        return {"available": False}
    last_t = int(s_now.index[-1])
    v_now = value_at(s_now, last_t)
    eps = [e for e in catalog.fountaining() if e["start_t"] <= now]
    out = {"available": True, "t": last_t, "value": v_now, "data_age_s": now - last_t}
    r = rate_series(s_now[s_now.index > last_t - 3 * 86400], 3.0)
    r24 = rate_series(s_now[s_now.index > last_t - 4 * 86400], 24.0)
    out["rate_3h"] = float(r.iloc[-1]) if len(r) else None
    out["rate_24h"] = float(r24.iloc[-1]) if len(r24) else None
    if eps:
        last = eps[-1]
        if last["end_t"] and last["end_t"] <= now:
            tr = trough(s_now, last["start_t"], last["end_t"])
            onset_v = value_at(s_now, last["start_t"])
            if tr is not None and v_now is not None:
                out["inflation_since_last_urad"] = v_now - tr
                out["last_deflation_urad"] = (onset_v - tr) if onset_v is not None else None
                if out["last_deflation_urad"] and out["last_deflation_urad"] > 0.5:
                    out["recovery_ratio"] = out["inflation_since_last_urad"] / out["last_deflation_urad"]
                out["gap_to_last_onset_urad"] = (v_now - onset_v) if onset_v is not None else None
            out["last_episode"] = last["label"]
            out["hours_since_last_end"] = (now - last["end_t"]) / 3600
        else:
            out["in_episode"] = True
            out["last_episode"] = last["label"]
    return out


def onset_thresholds(n_recent: int = 15) -> dict:
    rows = db.query(
        "SELECT label, num, onset_recovery_urad AS rec, onset_recovery_ratio AS ratio FROM episodes "
        "WHERE kind='fountaining' AND onset_recovery_urad IS NOT NULL ORDER BY start_t"
    )
    rows = rows[-n_recent:]
    if not rows:
        return {}
    rec = np.array([r["rec"] for r in rows], float)
    ratio = np.array([r["ratio"] for r in rows if r["ratio"] is not None], float)
    return {
        "n": len(rows),
        "episodes": [r["label"] for r in rows],
        "recovery_urad": {"p10": float(np.percentile(rec, 10)), "median": float(np.median(rec)),
                          "p90": float(np.percentile(rec, 90))},
        "recovery_ratio": ({"p10": float(np.percentile(ratio, 10)), "median": float(np.median(ratio)),
                            "p90": float(np.percentile(ratio, 90))} if len(ratio) else None),
    }
