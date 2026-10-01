"""RSAM tremor helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import db
from ..config import TREMOR_STA


def load(t0: int | None = None, t1: int | None = None) -> pd.Series:
    sql, p = "SELECT t, ums FROM rsam WHERE station=?", [TREMOR_STA]
    if t0 is not None:
        sql += " AND t>=?"; p.append(t0)
    if t1 is not None:
        sql += " AND t<=?"; p.append(t1)
    rows = db.query(sql + " ORDER BY t", p)
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series([r["ums"] for r in rows], index=np.array([r["t"] for r in rows], dtype=np.int64))


def rolling_features(s: pd.Series) -> pd.DataFrame:
    """1h and 24h medians (µm/s) on the 10-min grid; windows need ≥50% coverage."""
    if s.empty:
        return pd.DataFrame(columns=["r1h", "r24h", "r6h"])
    g = s.reindex(np.arange(s.index[0], s.index[-1] + 1, 600))
    return pd.DataFrame({
        "r1h": g.rolling(6, min_periods=3).median(),
        "r6h": g.rolling(36, min_periods=18).median(),
        "r24h": g.rolling(144, min_periods=72).median(),
    })


def current() -> dict:
    s = load()
    if s.empty:
        return {"available": False}
    f = rolling_features(s[s.index > s.index[-1] - 2 * 86400]).dropna(how="all")
    last = f.iloc[-1]
    return {
        "available": True, "t": int(s.index[-1]),
        "rsam_1h_ums": float(last["r1h"]) if pd.notna(last["r1h"]) else None,
        "rsam_24h_ums": float(last["r24h"]) if pd.notna(last["r24h"]) else None,
        "ratio_1h_24h": float(last["r1h"] / last["r24h"]) if pd.notna(last["r1h"]) and pd.notna(last["r24h"]) and last["r24h"] > 0 else None,
    }
