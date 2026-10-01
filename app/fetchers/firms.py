"""NASA FIRMS thermal hotspots over Halemaʻumaʻu (optional; needs FIRMS_MAP_KEY)."""
from __future__ import annotations

import csv
import hashlib
import io
from datetime import datetime, timezone

from .. import db
from ..config import FIRMS_MAP_KEY
from .base import get, tracked

API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{bbox}/{days}"
BBOX = "-155.32,19.39,-155.25,19.44"  # west,south,east,north around the summit caldera
SOURCES = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT"]


def _parse(text: str, source: str) -> list[tuple]:
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        try:
            hhmm = r["acq_time"].zfill(4)
            dt = datetime.strptime(f"{r['acq_date']} {hhmm}", "%Y-%m-%d %H%M").replace(tzinfo=timezone.utc)
            lat, lon = float(r["latitude"]), float(r["longitude"])
        except (KeyError, ValueError):
            continue
        bright = r.get("bright_ti4") or r.get("brightness")
        uid = hashlib.sha1(f"{source}{dt.isoformat()}{lat:.4f}{lon:.4f}".encode()).hexdigest()[:16]
        rows.append((uid, int(dt.timestamp()), lat, lon, float(r.get("frp") or 0),
                     float(bright) if bright else None, r.get("satellite"), r.get("confidence"), source))
    return rows


@tracked("firms")
def fetch(days: int = 2) -> str:
    if not FIRMS_MAP_KEY:
        raise RuntimeError("FIRMS_MAP_KEY not set (disabled)")
    total = 0
    for src in SOURCES:
        r = get(API.format(key=FIRMS_MAP_KEY, source=src, bbox=BBOX, days=days))
        r.raise_for_status()
        if "Invalid" in r.text[:200]:
            raise RuntimeError(r.text[:200])
        rows = _parse(r.text, src)
        with db.tx() as c:
            c.executemany("INSERT OR REPLACE INTO firms(id,t,lat,lon,frp,bright,satellite,confidence,source) "
                          "VALUES(?,?,?,?,?,?,?,?,?)", rows)
        total += len(rows)
    return f"{total} detections in last {days} d"
