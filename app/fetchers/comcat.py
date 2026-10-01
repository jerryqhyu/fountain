"""Earthquakes around Kīlauea from USGS ComCat (FDSN event service, GeoJSON)."""
from __future__ import annotations

import time
from datetime import datetime, timezone

from .. import db
from ..config import QUAKE_BOX, QUAKE_REGIONS
from .base import get, tracked

URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def region_of(lat: float, lon: float) -> str:
    for name, la0, la1, lo0, lo1 in QUAKE_REGIONS:
        if la0 <= lat <= la1 and lo0 <= lon <= lo1:
            return name
    return "other"


def _iso(t: int) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def pull(t0: int, t1: int) -> int:
    params = dict(format="geojson", starttime=_iso(t0), endtime=_iso(t1), orderby="time-asc",
                  limit=20000, **QUAKE_BOX)
    r = get(URL, params=params)
    if r.status_code == 204:
        return 0
    r.raise_for_status()
    feats = r.json().get("features", [])
    if len(feats) >= 20000:
        mid = (t0 + t1) // 2
        return pull(t0, mid) + pull(mid, t1)
    rows = []
    for f in feats:
        p, g = f["properties"], f["geometry"]["coordinates"]
        if p.get("type") not in (None, "earthquake"):
            continue
        lon, lat, depth = g[0], g[1], g[2]
        rows.append((f["id"], int(p["time"] // 1000), lat, lon, depth, p.get("mag"), p.get("magType"),
                     p.get("place"), region_of(lat, lon)))
    with db.tx() as c:
        c.executemany("INSERT OR REPLACE INTO earthquakes(id,t,lat,lon,depth,mag,magtype,place,region) "
                      "VALUES(?,?,?,?,?,?,?,?,?)", rows)
    return len(rows)


@tracked("comcat")
def fetch() -> str:
    now = int(time.time())
    # re-pull 2 days: catalog revisions (magnitudes, deletions of duplicates) settle quickly
    n = pull(now - 2 * 86400, now + 60)
    return f"{n} events in last 2 days"


def backfill(start: int = 1733011200) -> int:
    now = int(time.time())
    total, t = 0, start
    while t < now:
        t1 = min(t + 30 * 86400, now)
        total += pull(t, t1)
        t = t1
        time.sleep(0.5)
    return total
