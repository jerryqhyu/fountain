"""Stitch each data type's sources onto the 5-minute grid.

For every slot the first source in priority order that has a value wins (see
sources/__init__.py). Sources that are not directly comparable are mapped first:

tilt   - HVO plot digitizations carry an arbitrary per-plot offset: aligned by the median
         difference over the most recent 10 days of overlap with the series built so far
         (release → 3-month plot → 2-day plot, so the 2-day plot can align via the 3-month).
       - SDH fills remaining UWD gaps through UWD ≈ a·east + b·north + c fitted on the
         pause-time slots within 5 days of the gap; used only if R² ≥ 0.9.
tremor - 10-minute RSAM windows cover two slots each. UWE.QC is used as-is (~1% from UWE);
         OBL/UWB/RIMD are scaled by the median UWE/substitute ratio within 24 h of the gap.

The stitched series and the per-source mappings (offsets, fits, ratios) are stored so the
dashboard can show every source next to the authoritative series.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from . import db
from .config import GRID_S, GRID_START, RSAM_WINDOW_S
from .sources import SUBSTITUTE_TREMOR, TILT_SOURCES, TREMOR_SOURCES
from .sources.base import tracked

log = logging.getLogger("stitch")

TILT_TABLE = {s.key: s.table for s in TILT_SOURCES}
TREMOR_TABLE = {s.key: s.table for s in TREMOR_SOURCES}


# ------------------------------------------------------------------ grid helpers
def grid(t1: int | None = None) -> np.ndarray:
    t1 = int(t1 or time.time())
    return np.arange(GRID_START, (t1 // GRID_S) * GRID_S + 1, GRID_S, dtype=np.int64)


def on_grid(g: np.ndarray, t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Place values with slot-aligned timestamps onto the slot array g (NaN elsewhere)."""
    out = np.full(len(g), np.nan)
    if len(t) and len(g):
        idx = (np.asarray(t, np.int64) - int(g[0])) // GRID_S
        ok = (idx >= 0) & (idx < len(g))
        out[idx[ok]] = np.asarray(v, float)[ok]
    return out


def load(table: str, cols: str = "t, v") -> pd.DataFrame:
    rows = db.query(f"SELECT {cols} FROM {table} ORDER BY t")
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[c.strip() for c in cols.split(",")])


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Index ranges [i0, i1] where mask is True."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def episode_mask(g: np.ndarray) -> np.ndarray:
    m = np.zeros(len(g), bool)
    for e in db.query("SELECT start_t, end_t FROM episodes"):
        m |= (g >= e["start_t"]) & (g <= (e["end_t"] or 2**62))
    return m


def write_series(table: str, g: np.ndarray, v: np.ndarray, src: np.ndarray) -> int | None:
    """Upsert changed slots, delete slots that lost their value; return earliest changed slot."""
    old = load(table, "t, v, src")
    old_v = on_grid(g, old["t"].to_numpy(), old["v"].to_numpy()) if len(old) else np.full(len(g), np.nan)
    old_s = np.full(len(g), None, dtype=object)
    if len(old):
        idx = (old["t"].to_numpy(np.int64) - int(g[0])) // GRID_S
        ok = (idx >= 0) & (idx < len(g))
        old_s[idx[ok]] = old["src"].to_numpy()[ok]
    have, had = ~np.isnan(v), ~np.isnan(old_v)
    changed = (have != had) | (have & had & ((np.abs(v - np.nan_to_num(old_v)) > 1e-9) | (src != old_s)))
    if not changed.any():
        return None
    up = changed & have
    db.upsert(table, ["t", "v", "src"], zip(g[up].tolist(), v[up].tolist(), src[up].tolist()))
    gone = changed & ~have & had
    if gone.any():
        with db.tx() as c:
            c.executemany(f"DELETE FROM {table} WHERE t=?", [(int(t),) for t in g[gone]])
    return int(g[changed][0])


# ------------------------------------------------------------------ tilt
def _align(comp: np.ndarray, plot: np.ndarray, days: int = 10) -> float | None:
    both = np.where(~np.isnan(comp) & ~np.isnan(plot))[0]
    if len(both) < 24:
        return None
    both = both[both >= both[-1] - days * 86400 // GRID_S]
    return float(np.median(comp[both] - plot[both]))


def build_tilt(g: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    v = np.full(len(g), np.nan)
    src = np.full(len(g), None, dtype=object)
    info: dict = {"offsets": {}, "sdh_fits": []}

    rel = load(TILT_TABLE["uwd_release"])
    r = on_grid(g, rel["t"].to_numpy(), rel["v"].to_numpy())
    take = ~np.isnan(r)
    v[take], src[take] = r[take], "uwd_release"

    plots = {}
    for key in ("uwd_plot3m", "uwd_plot2d"):
        d = load(TILT_TABLE[key])
        plots[key] = on_grid(g, d["t"].to_numpy(), d["v"].to_numpy())
    # offsets: chain coarse → fine so the 2-day plot can align through the 3-month plot
    comp = v.copy()
    for key in ("uwd_plot3m", "uwd_plot2d"):
        off = _align(comp, plots[key])
        info["offsets"][key] = off
        if off is not None:
            fill = np.isnan(comp) & ~np.isnan(plots[key])
            comp[fill] = plots[key][fill] + off
    # fill: finer plot first
    for key in ("uwd_plot2d", "uwd_plot3m"):
        off = info["offsets"][key]
        if off is None:
            continue
        take = np.isnan(v) & ~np.isnan(plots[key])
        v[take], src[take] = plots[key][take] + off, key

    # SDH: local linear fit per remaining gap
    sdh = load(TILT_TABLE["sdh_release"], "t, east, north")
    if len(sdh):
        e = on_grid(g, sdh["t"].to_numpy(), sdh["east"].to_numpy())
        n = on_grid(g, sdh["t"].to_numpy(), sdh["north"].to_numpy())
        pause = ~episode_mask(g)
        first = np.where(~np.isnan(v))[0]
        last_rel = first[-1] if len(first) else -1
        ctx = 5 * 86400 // GRID_S
        for i0, i1 in runs(np.isnan(v)):
            if i0 == 0 or i1 >= last_rel:  # before the first / after the last UWD sample: nothing to fit
                continue
            lo, hi = max(0, i0 - ctx), min(len(g), i1 + 1 + ctx)
            win = np.zeros(len(g), bool)
            win[lo:hi] = True
            win[i0:i1 + 1] = False
            fitm = win & pause & ~np.isnan(v) & ~np.isnan(e) & ~np.isnan(n)
            gap = slice(i0, i1 + 1)
            if fitm.sum() < 200 or np.isnan(e[gap]).all():
                continue
            X = np.column_stack([e[fitm], n[fitm], np.ones(fitm.sum())])
            coef, *_ = np.linalg.lstsq(X, v[fitm], rcond=None)
            resid = v[fitm] - X @ coef
            r2 = 1 - resid.var() / v[fitm].var() if v[fitm].var() > 0 else 0.0
            fit = {"from": int(g[i0]), "to": int(g[i1]), "r2": round(float(r2), 3), "used": bool(r2 >= 0.9)}
            info["sdh_fits"].append(fit)
            if r2 >= 0.9:
                ok = np.zeros(len(g), bool)
                ok[gap] = ~np.isnan(e[gap]) & ~np.isnan(n[gap])
                v[ok] = coef[0] * e[ok] + coef[1] * n[ok] + coef[2]
                src[ok] = "sdh_release"
    return v, src, info


# ------------------------------------------------------------------ tremor
def windows_on_grid(g: np.ndarray, d: pd.DataFrame) -> np.ndarray:
    """Each 10-minute window value covers the slots it spans."""
    out = np.full(len(g), np.nan)
    if not len(d):
        return out
    t, val = d["t"].to_numpy(np.int64), d["v"].to_numpy(float)
    for k in range(RSAM_WINDOW_S // GRID_S):
        out = np.where(np.isnan(out), on_grid(g, t + k * GRID_S, val), out)
    return out


def build_tremor(g: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    v = np.full(len(g), np.nan)
    src = np.full(len(g), None, dtype=object)
    info: dict = {"ratios": []}
    for key in ("uwe", "uwe_qc"):
        s = windows_on_grid(g, load(TREMOR_TABLE[key]))
        take = np.isnan(v) & ~np.isnan(s)
        v[take], src[take] = s[take], key
    subs = {k: windows_on_grid(g, load(TREMOR_TABLE[k])) for k in SUBSTITUTE_TREMOR}
    primary = v.copy()
    ctx = 86400 // GRID_S
    for i0, i1 in runs(np.isnan(v)):
        lo, hi = max(0, i0 - ctx), min(len(g), i1 + 1 + ctx)
        for key in SUBSTITUTE_TREMOR:
            s = subs[key]
            gap = slice(i0, i1 + 1)
            if np.isnan(s[gap]).all():
                continue
            m = ~np.isnan(primary[lo:hi]) & ~np.isnan(s[lo:hi]) & (s[lo:hi] > 0)
            if m.sum() < 24:
                continue
            ratio = float(np.exp(np.median(np.log(primary[lo:hi][m] / s[lo:hi][m]))))
            take = np.zeros(len(g), bool)
            take[gap] = ~np.isnan(s[gap])
            v[take], src[take] = s[take] * ratio, key
            info["ratios"].append({"from": int(g[i0]), "to": int(g[i1]), "station": key, "ratio": round(ratio, 4)})
            break
    return v, src, info


# ------------------------------------------------------------------ entry point
@tracked("stitch")
def run() -> dict:
    g = grid()
    out = {}
    for name, builder, table in (("tilt", build_tilt, "series_tilt"), ("tremor", build_tremor, "series_rsam")):
        v, src, info = builder(g)
        changed = write_series(table, g, v, src)
        cover = {k: int((src == k).sum()) for k in set(src[src != None])}  # noqa: E711
        db.kv_set(f"stitch:{name}", {**info, "slots_by_source": cover, "missing": int(np.isnan(v).sum()),
                                     "grid": [int(g[0]), int(g[-1])], "built_at": int(time.time())})
        out[name] = changed
    return out
