"""USGS ScienceBase tilt releases (1-minute CSV) for UWD and SDH.

- 2024 release (covers December 2024)
- Jan–Jun 2025 release
- "July 1, 2025 to present with 60-day latency" dynamic release (monthly zips per station)

Stored as 5-minute means: UWD as the az-300 radial component (src_tilt_uwd_release.v),
SDH as east/north (src_tilt_sdh_release), since SDH is only used through a fitted mapping.
Downloaded zips are cached in data/tilt_release/ and never fetched twice.
"""
from __future__ import annotations

import io
import logging
import math
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


def zip_urls(station: str) -> list[tuple[str, str]]:
    out = []
    for item in FIXED_ITEMS:
        for f in _item_files(item):
            if f["name"] == f"{station}_digital.zip":
                out.append((f"{item}_{f['name']}", f["url"]))
    dyn = _dynamic_item(station)
    if dyn:
        for f in _item_files(dyn):
            if f["name"].startswith(f"{station}_") and f["name"].endswith(".zip"):
                out.append((f["name"], f["url"]))
    return out


def parse_zip(blob: bytes) -> pd.DataFrame:
    """East/north tilt as 5-minute means indexed by unix seconds."""
    frames = []
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in sorted(z.namelist()):
            if not name.lower().endswith(".csv") or "__MACOSX" in name:
                continue
            df = pd.read_csv(z.open(name), skipinitialspace=True)
            df.columns = [c.strip() for c in df.columns]
            tcol = next((c for c in df.columns if "time" in c.lower()), None)
            ecol = next((c for c in df.columns if c.lower().startswith("easttilt")), None)
            ncol = next((c for c in df.columns if c.lower().startswith("northtilt")), None)
            if not (tcol and ecol and ncol) or df.empty:
                continue
            d = pd.DataFrame({"east": df[ecol].astype(float).to_numpy(), "north": df[ncol].astype(float).to_numpy()},
                             index=pd.to_datetime(df[tcol], utc=True, format="ISO8601"))
            d = d[d.index >= pd.Timestamp(GRID_START, unit="s", tz="UTC")].dropna()
            if len(d):
                # label each 5-min mean by its bin start, matching the grid slot
                frames.append(d.resample("5min").mean().dropna())
    if not frames:
        return pd.DataFrame(columns=["east", "north"])
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index = out.index.as_unit("s").asi8.astype(int)
    return out


def az_component(east, north, az_deg: float = TILT_AZIMUTH_DEG):
    a = math.radians(az_deg)
    return np.asarray(east) * math.sin(a) + np.asarray(north) * math.cos(a)


def _sync(station: str) -> str:
    done = set(db.kv_get(f"release_done:{station}", []))
    new = 0
    for key, url in zip_urls(station):
        path = RELEASE_CACHE / key
        if key in done and path.exists():
            continue
        if not path.exists():
            r = get(url, timeout=300)
            r.raise_for_status()
            path.write_bytes(r.content)
        d = parse_zip(path.read_bytes())
        if station == "UWD":
            db.upsert("src_tilt_uwd_release", ["t", "v"],
                      zip(d.index.tolist(), az_component(d["east"], d["north"]).tolist()))
        else:
            db.upsert("src_tilt_sdh_release", ["t", "east", "north"],
                      zip(d.index.tolist(), d["east"].tolist(), d["north"].tolist()))
        done.add(key)
        new += 1
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
