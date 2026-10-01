"""Episode catalog.

Primary source: the episode table on USGS "Kīlauea Eruption Information"
(https://www.usgs.gov/volcanoes/kilauea/science/eruption-information), scraped daily.
A snapshot of that table ships in `seed_episodes.csv` so the app works offline and if
the page layout changes. Manual overrides go in `manual_episodes.csv` (same columns),
which always wins.

Rows whose episode number is not an integer (e.g. the Sept 14 2026 "--" row: new
vents erupting lava flows without fountaining) are stored with kind='non_fountaining'
and are excluded from onset statistics.
"""
from __future__ import annotations

import csv
import io
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from .. import db
from ..config import HST
from ..sources.base import get, tracked

log = logging.getLogger("episodes")

URL = "https://www.usgs.gov/volcanoes/kilauea/science/eruption-information"
HERE = Path(__file__).resolve().parent
SEED = HERE / "seed_episodes.csv"
MANUAL = HERE / "manual_episodes.csv"
COLUMNS = ["label", "start_hst", "end_hst", "fountain_height_m", "volume_mm3", "notes"]

_dt = re.compile(
    r"([A-Z][a-z]+)\s+(\d{1,2})(?:,\s*(\d{4}))?\s*[-–—]\s*(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s*m\.?", re.I
)
_noon = re.compile(r"([A-Z][a-z]+)\s+(\d{1,2})(?:,\s*(\d{4}))?\s*[-–—]\s*(noon|midnight)", re.I)


def parse_hst(s: str, default_year: int | None = None) -> int | None:
    """'January 24, 2025 - 11:28 p.m.' -> unix seconds. Year may be omitted if default given."""
    if not isinstance(s, str):
        return None
    s = s.replace("\xa0", " ").strip()
    m = _dt.search(s)
    if m:
        mon, day, year, hh, mm, ap = m.groups()
        year = year or default_year
        if year is None:
            return None
        h = int(hh) % 12 + (12 if ap.lower() == "p" else 0)
        dt = datetime.strptime(f"{mon} {day} {year}", "%B %d %Y").replace(
            hour=h, minute=int(mm or 0), tzinfo=HST
        )
        return int(dt.timestamp())
    m = _noon.search(s)
    if m:
        mon, day, year, which = m.groups()
        year = year or default_year
        if year is None:
            return None
        dt = datetime.strptime(f"{mon} {day} {year}", "%B %d %Y").replace(
            hour=12 if which.lower() == "noon" else 0, tzinfo=HST
        )
        return int(dt.timestamp())
    return None


def _num(x) -> float | None:
    try:
        m = re.search(r"[\d.]+", str(x))
        return float(m.group(0)) if m else None
    except Exception:
        return None


def scrape() -> pd.DataFrame:
    r = get(URL)
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))
    for t in tables:
        header = [str(x) for x in t.iloc[0].tolist()]
        if any("Episode" in h for h in header) and any("Start" in h for h in header):
            t = t.iloc[1:].copy()
            t.columns = header
            break
    else:
        raise ValueError("episode table not found on page")

    def col(key):
        return next(c for c in t.columns if key.lower() in c.lower())

    out = pd.DataFrame({
        "label": t[col("Episode")].astype(str).str.strip(),
        "start_hst": t[col("Start")].astype(str),
        "end_hst": t[col("Pause Date")].astype(str),
        "fountain_height_m": t[col("Fountain Height")].map(_num),
        "volume_mm3": t[col("volume")].map(_num),
        "notes": t[col("Notes")].astype(str).replace("nan", ""),
    })
    # A non-integer label (e.g. "--") gets a stable name from its start date.
    def fix_label(row):
        if re.fullmatch(r"\d+", row["label"]):
            return row["label"]
        ts = parse_hst(row["start_hst"])
        return "vents-" + (datetime.fromtimestamp(ts, HST).strftime("%Y%m%d") if ts else "unknown")
    out["label"] = out.apply(fix_label, axis=1)
    if len(out) < 40:
        raise ValueError(f"episode table suspiciously short ({len(out)} rows)")
    return out


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype={"label": str}).fillna({"notes": ""})


def upsert(df: pd.DataFrame, source: str) -> int:
    now = int(time.time())
    rows = []
    for _, r in df.iterrows():
        label = str(r["label"]).strip()
        start = parse_hst(r["start_hst"])
        if start is None:
            continue
        end = parse_hst(r["end_hst"], default_year=datetime.fromtimestamp(start, HST).year)
        num = int(label) if label.isdigit() else None
        kind = "fountaining" if num is not None else "non_fountaining"
        fh = r.get("fountain_height_m")
        vol = r.get("volume_mm3")
        rows.append((
            label, num, kind, start, end,
            None if pd.isna(fh) else float(fh), None if pd.isna(vol) else float(vol),
            str(r.get("notes") or ""), source, now,
        ))
    with db.tx() as c:
        c.executemany(
            """INSERT INTO episodes(label,num,kind,start_t,end_t,fountain_height_m,volume_mm3,notes,source,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(label) DO UPDATE SET num=excluded.num, kind=excluded.kind,
                 start_t=excluded.start_t, end_t=COALESCE(excluded.end_t, episodes.end_t),
                 fountain_height_m=excluded.fountain_height_m, volume_mm3=excluded.volume_mm3,
                 notes=excluded.notes, source=excluded.source, updated_at=excluded.updated_at""",
            rows,
        )
    return len(rows)


def seed() -> None:
    """Load the shipped snapshot (only fills gaps) and manual overrides."""
    have = {r["label"] for r in db.query("SELECT label FROM episodes")}
    s = _load_csv(SEED)
    s = s[~s["label"].isin(have)]
    if len(s):
        upsert(s, "seed")
    m = _load_csv(MANUAL)
    if len(m):
        upsert(m, "manual")


@tracked("usgs_episode_table")
def fetch() -> str:
    df = scrape()
    n = upsert(df, "usgs_table")
    m = _load_csv(MANUAL)
    if len(m):
        upsert(m, "manual")
    # keep the shipped snapshot fresh so a later outage still has recent rows
    df.to_csv(SEED, index=False, quoting=csv.QUOTE_MINIMAL)
    return f"{n} rows"


def catalog(include_non_fountaining: bool = True) -> list[dict]:
    rows = db.query("SELECT * FROM episodes ORDER BY start_t")
    if not include_non_fountaining:
        rows = [r for r in rows if r["kind"] == "fountaining"]
    fountain = [r for r in rows if r["kind"] == "fountaining"]
    for r in rows:
        r["duration_h"] = (r["end_t"] - r["start_t"]) / 3600 if r["end_t"] else None
    for i, r in enumerate(fountain):
        r["repose_before_h"] = (
            (r["start_t"] - fountain[i - 1]["end_t"]) / 3600 if i > 0 and fountain[i - 1]["end_t"] else None
        )
        r["onset_interval_h"] = (r["start_t"] - fountain[i - 1]["start_t"]) / 3600 if i > 0 else None
    return rows


def fountaining() -> list[dict]:
    return [r for r in catalog() if r["kind"] == "fountaining"]


def last_fountaining() -> dict | None:
    f = fountaining()
    return f[-1] if f else None
