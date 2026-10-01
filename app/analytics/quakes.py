"""Earthquake counts and rates by region and depth."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .. import db
from ..config import HST

DEPTH_BINS = [(-10, 5, "shallow (<5 km)"), (5, 20, "intermediate (5–20 km)"), (20, 1000, "deep (>20 km)")]


def load(t0: int, t1: int | None = None) -> pd.DataFrame:
    t1 = t1 or int(time.time()) + 60
    return pd.DataFrame(db.query(
        "SELECT id, t, lat, lon, depth, mag, magtype, place, region FROM earthquakes WHERE t>=? AND t<? ORDER BY t",
        (t0, t1)))


def depth_class(d: float) -> str:
    for lo, hi, name in DEPTH_BINS:
        if lo <= d < hi:
            return name
    return "deep (>20 km)"


def summary(days: int = 7) -> dict:
    now = int(time.time())
    df = load(now - days * 86400)
    if df.empty:
        return {"events": [], "daily": [], "counts": {}}
    df["depth_class"] = df["depth"].map(depth_class)
    df["day"] = pd.to_datetime(df["t"], unit="s", utc=True).dt.tz_convert(HST).dt.strftime("%Y-%m-%d")
    daily = df.groupby(["day", "region"]).size().unstack(fill_value=0).reset_index()
    daily_depth = df.groupby(["day", "depth_class"]).size().unstack(fill_value=0).reset_index()
    last24 = df[df["t"] >= now - 86400]
    counts = {
        "total": int(len(df)),
        "last_24h": int(len(last24)),
        "summit_last_24h": int((last24["region"] == "summit").sum()),
        "by_region": df["region"].value_counts().to_dict(),
        "by_depth": df["depth_class"].value_counts().to_dict(),
        "max_mag": float(df["mag"].max()) if df["mag"].notna().any() else None,
    }
    ev = df[["id", "t", "lat", "lon", "depth", "mag", "place", "region", "depth_class"]]
    return {
        "events": ev.replace({np.nan: None}).to_dict("records"),
        "daily": daily.to_dict("records"),
        "daily_depth": daily_depth.to_dict("records"),
        "counts": counts,
    }


def count_series(times: np.ndarray, region: str | None, window_s: int) -> np.ndarray:
    """Number of events in (t - window, t] for each t (used for model features)."""
    sql = "SELECT t FROM earthquakes" + (" WHERE region=?" if region else "") + " ORDER BY t"
    ev = np.array([r["t"] for r in db.query(sql, (region,) if region else ())], dtype=np.int64)
    if len(ev) == 0:
        return np.zeros(len(times))
    return np.searchsorted(ev, times, side="right") - np.searchsorted(ev, times - window_s, side="right")
