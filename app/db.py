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

SCHEMA = """
CREATE TABLE IF NOT EXISTS source_health (
    source TEXT PRIMARY KEY,
    last_attempt INTEGER,
    last_success INTEGER,
    last_error TEXT,
    last_error_at INTEGER,
    detail TEXT
);
CREATE TABLE IF NOT EXISTS raw_cache (
    key TEXT PRIMARY KEY,
    fetched_at INTEGER,
    body TEXT
);
CREATE TABLE IF NOT EXISTS status_history (
    fetched_at INTEGER PRIMARY KEY,
    alert_level TEXT,
    color_code TEXT,
    alert_date TEXT,
    notice_id TEXT,
    synopsis TEXT
);
CREATE TABLE IF NOT EXISTS notices (
    notice_id TEXT PRIMARY KEY,
    sent_unix INTEGER,
    type_cd TEXT,
    type_title TEXT,
    title TEXT,
    synopsis TEXT,
    text TEXT,
    html TEXT,
    url TEXT
);
CREATE INDEX IF NOT EXISTS notices_sent ON notices(sent_unix);
CREATE TABLE IF NOT EXISTS earthquakes (
    id TEXT PRIMARY KEY,
    t INTEGER,
    lat REAL, lon REAL, depth REAL, mag REAL, magtype TEXT,
    place TEXT, region TEXT
);
CREATE INDEX IF NOT EXISTS eq_t ON earthquakes(t);
CREATE TABLE IF NOT EXISTS tilt (
    t INTEGER PRIMARY KEY,
    v REAL,
    src TEXT
);
CREATE TABLE IF NOT EXISTS tilt_plot (
    plot TEXT,
    t INTEGER,
    v REAL, vmin REAL, vmax REAL,
    fetched_at INTEGER,
    PRIMARY KEY (plot, t)
);
CREATE TABLE IF NOT EXISTS rsam (
    station TEXT,
    t INTEGER,
    counts REAL,
    ums REAL,
    PRIMARY KEY (station, t)
);
CREATE TABLE IF NOT EXISTS episodes (
    label TEXT PRIMARY KEY,
    num INTEGER,
    kind TEXT,
    start_t INTEGER,
    end_t INTEGER,
    fountain_height_m REAL,
    volume_mm3 REAL,
    notes TEXT,
    source TEXT,
    deflation_urad REAL,
    onset_recovery_urad REAL,
    onset_recovery_ratio REAL,
    updated_at INTEGER
);
CREATE TABLE IF NOT EXISTS episode_suggestions (
    notice_id TEXT PRIMARY KEY,
    sent_unix INTEGER,
    keywords TEXT,
    episode_num INTEGER,
    phase TEXT,
    snippet TEXT,
    status TEXT DEFAULT 'pending'
);
CREATE TABLE IF NOT EXISTS firms (
    id TEXT PRIMARY KEY,
    t INTEGER,
    lat REAL, lon REAL, frp REAL, bright REAL,
    satellite TEXT, confidence TEXT, source TEXT
);
CREATE INDEX IF NOT EXISTS firms_t ON firms(t);
CREATE TABLE IF NOT EXISTS probability_log (
    t INTEGER,
    model TEXT,
    version TEXT,
    p12 REAL, p24 REAL, p72 REAL,
    payload TEXT,
    PRIMARY KEY (t, model)
);
-- values that were shown live by a model that has since been replaced (kept for auditing)
CREATE TABLE IF NOT EXISTS probability_archive (
    t INTEGER,
    model TEXT,
    version TEXT,
    trained_at INTEGER,
    p12 REAL, p24 REAL, p72 REAL,
    payload TEXT,
    PRIMARY KEY (t, model, trained_at)
);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at INTEGER
);
"""


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
