"""SQLite storage. One short-lived connection per call; WAL lets the scheduler and API share it."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterable

from .config import DB_PATH

_write_lock = threading.Lock()

# Time-series source tables: one per source, at the source's native resolution.
SERIES_SOURCE_TABLES = {
    # tilt (µrad). UWD values are the az-300 radial component.
    "src_tilt_uwd_release": "t INTEGER PRIMARY KEY, v REAL",            # USGS 1-min CSV, 5-min means
    "src_tilt_uwd_plot2d": "t INTEGER PRIMARY KEY, v REAL, fetched_at INTEGER",  # digitized, plot's own offset
    "src_tilt_uwd_plot3m": "t INTEGER PRIMARY KEY, v REAL, fetched_at INTEGER",
    "src_tilt_sdh_release": "t INTEGER PRIMARY KEY, east REAL, north REAL",  # USGS 1-min CSV, 5-min means
    # tremor: RSAM in µm/s, 10-minute windows keyed by window start
    "src_rsam_uwe": "t INTEGER PRIMARY KEY, v REAL",
    "src_rsam_uwe_qc": "t INTEGER PRIMARY KEY, v REAL",
    "src_rsam_obl": "t INTEGER PRIMARY KEY, v REAL",
    "src_rsam_uwb": "t INTEGER PRIMARY KEY, v REAL",
    "src_rsam_rimd": "t INTEGER PRIMARY KEY, v REAL",
}

# Feature frame columns (one row per 5-minute grid slot).
FRAME_COLUMNS = [
    "in_episode", "last_label", "last_end", "next_onset",
    "hours_since_end", "recovery_ratio", "inflation_urad", "gap_to_onset_urad", "last_deflation_urad",
    "tilt_rate_6h", "tilt_rate_24h", "tilt_value",
    "rsam_log", "rsam_ratio_log", "rsam_trend_log", "rsam_1h_ums",
    "eq_summit_24h", "eq_all_24h", "precursor",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS source_health (
    source TEXT PRIMARY KEY,
    last_attempt INTEGER,
    last_success INTEGER,
    last_error TEXT,
    last_error_at INTEGER,
    detail TEXT
);
CREATE TABLE IF NOT EXISTS raw_cache (key TEXT PRIMARY KEY, fetched_at INTEGER, body TEXT);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT, updated_at INTEGER);

-- event/document sources
CREATE TABLE IF NOT EXISTS notices (
    notice_id TEXT PRIMARY KEY, sent_unix INTEGER, type_cd TEXT, type_title TEXT,
    title TEXT, synopsis TEXT, text TEXT, html TEXT, url TEXT
);
CREATE INDEX IF NOT EXISTS notices_sent ON notices(sent_unix);
CREATE TABLE IF NOT EXISTS earthquakes (
    id TEXT PRIMARY KEY, t INTEGER, lat REAL, lon REAL, depth REAL, mag REAL, magtype TEXT,
    place TEXT, region TEXT
);
CREATE INDEX IF NOT EXISTS eq_t ON earthquakes(t);
CREATE TABLE IF NOT EXISTS episodes (
    label TEXT PRIMARY KEY, num INTEGER, kind TEXT, start_t INTEGER, end_t INTEGER,
    fountain_height_m REAL, volume_mm3 REAL, notes TEXT, source TEXT,
    deflation_urad REAL, onset_recovery_urad REAL, onset_recovery_ratio REAL, updated_at INTEGER
);
CREATE TABLE IF NOT EXISTS episode_suggestions (
    notice_id TEXT PRIMARY KEY, sent_unix INTEGER, keywords TEXT, episode_num INTEGER,
    phase TEXT, snippet TEXT, status TEXT DEFAULT 'pending'
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY, t INTEGER, kind TEXT, text TEXT, truncated INTEGER
);
CREATE INDEX IF NOT EXISTS messages_t ON messages(t);
CREATE TABLE IF NOT EXISTS firms (
    id TEXT PRIMARY KEY, t INTEGER, lat REAL, lon REAL, frp REAL, bright REAL,
    satellite TEXT, confidence TEXT, source TEXT
);
CREATE INDEX IF NOT EXISTS firms_t ON firms(t);

-- stitched series on the 5-minute grid (src = which source supplied the slot)
CREATE TABLE IF NOT EXISTS series_tilt (t INTEGER PRIMARY KEY, v REAL, src TEXT);
CREATE TABLE IF NOT EXISTS series_rsam (t INTEGER PRIMARY KEY, v REAL, src TEXT);

-- model outputs on the grid; NULL where an input is genuinely missing or an episode is under way
CREATE TABLE IF NOT EXISTS pred_hazard (t INTEGER PRIMARY KEY, p12 REAL, p24 REAL, p72 REAL);
CREATE TABLE IF NOT EXISTS pred_ml (t INTEGER PRIMARY KEY, p12 REAL, p24 REAL, p72 REAL);
""" + "".join(f"CREATE TABLE IF NOT EXISTS {name} ({cols});\n" for name, cols in SERIES_SOURCE_TABLES.items()) + (
    "CREATE TABLE IF NOT EXISTS frame (t INTEGER PRIMARY KEY, "
    + ", ".join(f"{c} {'TEXT' if c == 'last_label' else 'REAL'}" for c in FRAME_COLUMNS) + ");\n"
)


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init() -> None:
    with connect() as c:
        c.executescript(SCHEMA)


@contextmanager
def tx():
    """Serialized write transaction."""
    with _write_lock:
        conn = connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def query(sql: str, params: Iterable[Any] = ()) -> list[dict]:
    conn = connect()
    try:
        return [dict(r) for r in conn.execute(sql, tuple(params)).fetchall()]
    finally:
        conn.close()


def query_one(sql: str, params: Iterable[Any] = ()) -> dict | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def kv_get(key: str, default: Any = None) -> Any:
    row = query_one("SELECT value FROM kv WHERE key=?", (key,))
    return json.loads(row["value"]) if row else default


def kv_set(key: str, value: Any) -> None:
    with tx() as c:
        c.execute(
            "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, json.dumps(value, default=str), int(time.time())),
        )


def cache_put(key: str, body: str) -> None:
    with tx() as c:
        c.execute(
            "INSERT OR REPLACE INTO raw_cache(key,fetched_at,body) VALUES(?,?,?)",
            (key, int(time.time()), body),
        )


def cache_get(key: str) -> tuple[int, str] | None:
    row = query_one("SELECT fetched_at, body FROM raw_cache WHERE key=?", (key,))
    return (row["fetched_at"], row["body"]) if row else None


def upsert(table: str, columns: list[str], rows: Iterable[tuple]) -> int:
    rows = list(rows)
    if rows:
        ph = ",".join("?" * len(columns))
        with tx() as c:
            c.executemany(f"INSERT OR REPLACE INTO {table}({','.join(columns)}) VALUES({ph})", rows)
    return len(rows)
