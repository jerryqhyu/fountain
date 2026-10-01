"""Historical UWD tilt from USGS ScienceBase data releases (1-minute CSV).

- 2024 release (covers Dec 2024, episodes 1–3)
- Jan–Jun 2025 release
- "July 1, 2025 to present with 60-day latency" dynamic release (monthly zips)

Values are projected onto azimuth 300° (the radial component HVO plots) and averaged
to 10-minute bins. Stored in `tilt` with src='release:<file>'. Separate CSV files within
one zip mark instrument relevels; a `segment` id is kept via the src string.
"""
from __future__ import annotations

import io
import logging
import math
import zipfile

import numpy as np
import pandas as pd

from .. import db
from ..config import DATA_DIR, TILT_AZIMUTH_DEG
from .base import get, tracked

log = logging.getLogger("tilt_release")

SB = "https://www.sciencebase.gov/catalog"
FIXED_ITEMS = ["67bfbc28d34e8876fcbfca43", "67ead922d34ed02007f83585"]  # 2024, 2025H1
DYNAMIC_PARENT = "68ae1fb1d4be025fdd4ae0df"
CACHE = DATA_DIR / "tilt_release"
CACHE.mkdir(exist_ok=True)
FROM_T = 1733011200  # 2024-12-01


def _item_files(item_id: str) -> list[dict]:
    r = get(f"{SB}/item/{item_id}", params={"format": "json", "fields": "title,files"})
    r.raise_for_status()
    return r.json().get("files", []) or []


def _uwd_dynamic_item() -> str | None:
    r = get(f"{SB}/items", params={"parentId": DYNAMIC_PARENT, "format": "json", "fields": "title", "max": 100})
    r.raise_for_status()
    for it in r.json().get("items", []):
        if it["title"].rstrip().split(":")[-1].strip().startswith("UWD"):
            return it["id"]
    return None


def list_zip_urls() -> list[tuple[str, str]]:
    out = []
    for item in FIXED_ITEMS:
        for f in _item_files(item):
            if f["name"] == "UWD_digital.zip":
                out.append((f"{item}_{f['name']}", f["url"]))
    dyn = _uwd_dynamic_item()
    if dyn:
        for f in _item_files(dyn):
            if f["name"].startswith("UWD_") and f["name"].endswith(".zip"):
                out.append((f["name"], f["url"]))
    return out


def az_component(east: np.ndarray, north: np.ndarray, az_deg: float = TILT_AZIMUTH_DEG) -> np.ndarray:
    a = math.radians(az_deg)
    return east * math.sin(a) + north * math.cos(a)


def parse_zip(blob: bytes) -> list[pd.DataFrame]:
    frames = []
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in sorted(z.namelist()):
            if not name.lower().endswith(".csv") or "__MACOSX" in name:
                continue
            df = pd.read_csv(z.open(name), skipinitialspace=True)
            df.columns = [c.strip() for c in df.columns]
            tcol = [c for c in df.columns if "time" in c.lower()][0]
            ecol = [c for c in df.columns if c.lower().startswith("easttilt")][0]
            ncol = [c for c in df.columns if c.lower().startswith("northtilt")][0]
            t = pd.to_datetime(df[tcol], utc=True, format="ISO8601")
            v = az_component(df[ecol].to_numpy(float), df[ncol].to_numpy(float))
            s = pd.Series(v, index=t).dropna()
            s = s[s.index >= pd.Timestamp(FROM_T, unit="s", tz="UTC")]
            if len(s):
                frames.append(pd.DataFrame({"v": s, "file": name.split("/")[-1]}))
    return frames


def to_bins(df: pd.DataFrame) -> pd.DataFrame:
    b = df["v"].resample("10min").mean().dropna()
    return pd.DataFrame({"t": b.index.as_unit("s").asi8.astype(int), "v": b.to_numpy()})


@tracked("tilt_release")
def fetch(force: bool = False) -> str:
    done = set(db.kv_get("tilt_release_done", []))
    new = 0
    for key, url in list_zip_urls():
        path = CACHE / key
        # The newest monthly file may be re-issued; refresh anything from the last 3 months.
        if key in done and path.exists() and not force:
            continue
        r = get(url, timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
        rows = []
        for df in parse_zip(r.content):
            b = to_bins(df)
            src = f"release:{df['file'].iloc[0]}"
            rows += [(int(t), float(v), src) for t, v in zip(b["t"], b["v"])]
        with db.tx() as c:
            c.executemany("INSERT OR REPLACE INTO tilt(t,v,src) VALUES(?,?,?)", rows)
        done.add(key)
        new += 1
        log.info("tilt release %s: %d bins", key, len(rows))
    db.kv_set("tilt_release_done", sorted(done))
    last = db.query_one("SELECT MAX(t) AS t FROM tilt WHERE src LIKE 'release:%'")
    return f"{new} new files; release data through {last['t'] if last else None}"
