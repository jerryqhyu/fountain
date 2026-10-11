"""USGS ScienceBase tilt releases (1-minute CSV) for UWD and SDH.

- 2024 release (covers December 2024)
- Jan–Jun 2025 release
- "July 1, 2025 to present with 60-day latency" dynamic release (monthly zips per station)

Stored as 5-minute means: UWD as the az-300 radial component (src_tilt_uwd_release.v),
SDH as east/north (src_tilt_sdh_release), since SDH is only used through a fitted mapping.
Downloaded zips are cached in data/tilt_release/ and fetched again only when ScienceBase lists a
different file size (USGS revises monthly files).

USGS starts a new CSV when a tiltmeter is re-levelled: after a data gap the east/north readings
resume on a different, arbitrary baseline. The stored samples stay exactly as published; those
file boundaries are recorded (kv `release_breaks:<station>`) and removed when stitching.
"""
from __future__ import annotations

import io
import logging
import math
import re
import zipfile

import numpy as np
import pandas as pd

from .. import db
from ..config import GRID_START, RELEASE_CACHE, TILT_AZIMUTH_DEG
from .base import get, tracked

log = logging.getLogger("tilt_release")

SB = "https://www.sciencebase.gov/catalog"
FIXED_ITEMS = ["67bfbc28d34e8876fcbfca43", "67ead922d34ed02007f83585"]  # 2024, 2025H1
DYNAMIC_PARENT = "68ae1fb1d4be025fdd4ae0df"
COLUMNS = ["UTCDateTime", "Xtilt(microrad)", "Ytilt(microrad)", "EastTilt(microrad)", "NorthTilt(microrad)",
           "HoleTemp(celsius)", "BoxTemp(celsius)", "Voltage(V)"]
BREAK_GAP_S = 600    # a new file that starts after a data gap this long ...
RELEVEL_URAD = 5.0   # ... with east or north offset by more than this is a re-levelled instrument


def _item_files(item_id: str) -> list[dict]:
    r = get(f"{SB}/item/{item_id}", params={"format": "json", "fields": "title,files"})
    r.raise_for_status()
    return r.json().get("files", []) or []


def _dynamic_item(station: str) -> str | None:
    r = get(f"{SB}/items", params={"parentId": DYNAMIC_PARENT, "format": "json", "fields": "title", "max": 100})
    r.raise_for_status()
    for it in r.json().get("items", []):
        if it["title"].rstrip().split(":")[-1].strip().startswith(station):
            return it["id"]
    return None


def zip_urls(station: str) -> list[tuple[str, str, int | None]]:
    """(cache name, download url, size listed by ScienceBase) of every release file for a station."""
    out = []
    for item in FIXED_ITEMS:
        for f in _item_files(item):
            if f["name"] == f"{station}_digital.zip":
                out.append((f"{item}_{f['name']}", f["url"], f.get("size")))
    dyn = _dynamic_item(station)
    if dyn:
        for f in _item_files(dyn):
            if f["name"].startswith(f"{station}_") and f["name"].endswith(".zip"):
                out.append((f["name"], f["url"], f.get("size")))
    return out


def read_zip(blob: bytes) -> list[pd.DataFrame]:
    """East/north tilt at the published 1-minute sampling: one frame per CSV, in time order."""
    frames = []
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in sorted(z.namelist()):
            if not name.lower().endswith(".csv") or "__MACOSX" in name:
                continue
            df = pd.read_csv(z.open(name), skipinitialspace=True)
            df.columns = [c.strip() for c in df.columns]
            if len(df.columns) == len(COLUMNS) and re.match(r"\d{4}-\d\d-\d\dT", df.columns[0]):
                # published without its header row (SDH, Aug-Dec 2024): the first sample was read as column names
                df = pd.read_csv(z.open(name), skipinitialspace=True, header=None, names=COLUMNS)
            tcol = next((c for c in df.columns if "time" in c.lower()), None)
            ecol = next((c for c in df.columns if c.lower().startswith("easttilt")), None)
            ncol = next((c for c in df.columns if c.lower().startswith("northtilt")), None)
            if not (tcol and ecol and ncol) or df.empty:
                continue
            d = pd.DataFrame({"east": df[ecol].astype(float).to_numpy(), "north": df[ncol].astype(float).to_numpy()},
                             index=pd.to_datetime(df[tcol], utc=True, format="ISO8601"))
            d = d[d.index >= pd.Timestamp(GRID_START, unit="s", tz="UTC")].dropna()
            if len(d):
                frames.append(d)
    return sorted(frames, key=lambda d: d.index[0])


def find_breaks(frames: list[pd.DataFrame]) -> list[dict]:
    """Boundaries between consecutive CSVs where the instrument was re-levelled during a data gap."""
    out = []
    for a, b in zip(frames, frames[1:]):
        if (b.index[0] - a.index[-1]).total_seconds() <= BREAK_GAP_S:
            continue
        d = (b[b.index < b.index[0] + pd.Timedelta(minutes=30)].median()
             - a[a.index > a.index[-1] - pd.Timedelta(minutes=30)].median())
        if max(abs(d["east"]), abs(d["north"])) > RELEVEL_URAD:
            out.append({"from": int(a.index[-1].timestamp()), "to": int(b.index[0].timestamp()),
                        "east": round(float(d["east"]), 2), "north": round(float(d["north"]), 2)})
    return out


def means(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """East/north tilt as 5-minute means indexed by unix seconds (each mean labelled by its bin start)."""
    frames = [d.resample("5min").mean().dropna() for d in frames]
    if not frames:
        return pd.DataFrame(columns=["east", "north"])
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index = out.index.as_unit("s").asi8.astype(int)
    return out


def parse_zip(blob: bytes) -> pd.DataFrame:
    return means(read_zip(blob))


def az_component(east, north, az_deg: float = TILT_AZIMUTH_DEG):
    a = math.radians(az_deg)
    return np.asarray(east) * math.sin(a) + np.asarray(north) * math.cos(a)


def _sync(station: str) -> str:
    known = db.kv_get(f"release_breaks:{station}")
    # no break list yet: every cached file was read by an earlier version, so read them all again
    done = set(db.kv_get(f"release_done:{station}", [])) if known is not None else set()
    breaks = {b["to"]: b for b in known or []}
    new = 0
    for key, url, size in zip_urls(station):
        path = RELEASE_CACHE / key
        revised = path.exists() and size is not None and path.stat().st_size != size
        if key in done and path.exists() and not revised:
            continue
        if not path.exists() or revised:
            r = get(url, timeout=300)
            r.raise_for_status()
            path.write_bytes(r.content)
        frames = read_zip(path.read_bytes())
        breaks.update({b["to"]: b for b in find_breaks(frames)})
        d = means(frames)
        if station == "UWD":
            db.upsert("src_tilt_uwd_release", ["t", "v"],
                      zip(d.index.tolist(), az_component(d["east"], d["north"]).tolist()))
        else:
            db.upsert("src_tilt_sdh_release", ["t", "east", "north"],
                      zip(d.index.tolist(), d["east"].tolist(), d["north"].tolist()))
        done.add(key)
        new += 1
    db.kv_set(f"release_breaks:{station}", [breaks[k] for k in sorted(breaks)])
    db.kv_set(f"release_done:{station}", sorted(done))
    table = "src_tilt_uwd_release" if station == "UWD" else "src_tilt_sdh_release"
    last = db.query_one(f"SELECT MAX(t) AS t FROM {table}")["t"]
    return f"{new} new files; data through {last}"


@tracked("uwd_release")
def fetch_uwd() -> str:
    return _sync("UWD")


@tracked("sdh_release")
def fetch_sdh() -> str:
    return _sync("SDH")
