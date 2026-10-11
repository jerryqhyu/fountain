"""Fill an empty database: import what an older database already fetched, download the rest.

    uv run python -m app.bootstrap

Every step is idempotent, so the server reruns it on start until one run has reached the end:
a restart in the middle resumes where it stopped. Without data/volcano.db everything is
downloaded (≈1 h, mostly UWE tremor history from EarthScope).
"""
from __future__ import annotations

import logging
import sqlite3
import time

import numpy as np

from . import db, pipeline
from .config import DATA_DIR, GRID_START
from .episodes import catalog
from .sources import comcat, firms, hans, hvo_messages, rsam, tilt_plots, tilt_release, usgs_status, weather

log = logging.getLogger("bootstrap")
LEGACY_DB = DATA_DIR / "volcano.db"
# In the legacy DB, UWE.QC windows were merged into UWE during this primary-sensor outage.
LEGACY_QC_WINDOW = (1786833000, 1786833000 + 745 * 3600)
REPAIRS = 1  # bump when repair() gains a step


def _legacy() -> sqlite3.Connection | None:
    if not LEGACY_DB.exists():
        return None
    c = sqlite3.connect(LEGACY_DB)
    c.row_factory = sqlite3.Row
    return c


def import_legacy() -> None:
    c = _legacy()
    if c is None:
        return
    for table in ("notices", "earthquakes", "episodes", "episode_suggestions", "firms"):
        rows = [dict(r) for r in c.execute(f"SELECT * FROM {table}")]
        if rows:
            cols = list(rows[0].keys())
            db.upsert(table, cols, [tuple(r[k] for k in cols) for r in rows])
            log.info("imported %d %s", len(rows), table)
    # measured UWE RSAM (gap-fill rows excluded; QC-merged outage excluded and recomputed as UWE.QC)
    q0, q1 = LEGACY_QC_WINDOW
    rows = c.execute("SELECT t, ums FROM rsam WHERE station='UWE' AND src IS NULL AND NOT (t BETWEEN ? AND ?)",
                     (q0, q1)).fetchall()
    db.upsert("src_rsam_uwe", ["t", "v"], [(r["t"], r["ums"]) for r in rows])
    log.info("imported %d UWE RSAM windows", len(rows))
    # earlier HVO plot digitizations: stored offset-aligned at 10-min, convert back to each
    # plot's own offset and interpolate onto 5-minute samples
    stitch_info = {r["key"]: r["value"] for r in c.execute("SELECT key, value FROM kv WHERE key='tilt_stitch'")}
    import json
    layers = json.loads(stitch_info.get("tilt_stitch", "{}")).get("layers", {})
    groups = {"src_tilt_uwd_plot2d": (["plot:2day", "plot:week", "plot:month"], layers.get("2day", {}).get("offset")),
              "src_tilt_uwd_plot3m": (["plot:3month", "plot:wb_3month_20260806"], layers.get("3month", {}).get("offset"))}
    now = int(time.time())
    for table, (srcs, off) in groups.items():
        if off is None:
            continue
        ph = ",".join("?" * len(srcs))
        rows = c.execute(f"SELECT t, v FROM tilt WHERE src IN ({ph}) ORDER BY t", srcs).fetchall()
        if not rows:
            continue
        t = np.array([r["t"] for r in rows], np.int64)
        v = np.array([r["v"] for r in rows], float) - off
        g = np.arange(int(np.ceil(t[0] / tilt_plots.SAMPLE_S)) * tilt_plots.SAMPLE_S, int(t[-1]) + 1, tilt_plots.SAMPLE_S)
        gv = np.interp(g, t, v)
        i = np.clip(np.searchsorted(t, g), 1, len(t) - 1)
        ok = (t[i] - t[i - 1]) <= 1200
        db.upsert(table, ["t", "v", "fetched_at"], [(int(a), float(b), now) for a, b in zip(g[ok], gv[ok])])
        log.info("imported %d slots into %s", int(ok.sum()), table)


def step(name, fn, *a):
    log.info("bootstrap: %s", name)
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001
        log.warning("bootstrap step %s failed: %s", name, e)


def repair() -> None:
    """One-off corrections to rows stored by earlier versions; idempotent, rerun on start until finished."""
    now = int(time.time())
    step("ComCat revisions", comcat.backfill)
    step("UWD release breaks", tilt_release.fetch_uwd)
    step("SDH release, headerless file", tilt_release.fetch_sdh)
    for key in rsam.TABLE:
        step(f"{key} chunk-edge windows", rsam.repair_edges, key)
    # live windows were dimmed by the same taper; polling began when the first sensitivity was cached
    first = db.query_one("SELECT MIN(updated_at) AS t FROM kv WHERE key LIKE 'sens:%'")["t"] or now
    live_since = first - 6 * 3600
    for key in ("uwe", "uwe_qc"):
        step(f"{key} live windows", rsam.compute_range, key, live_since, now, 0.3, True)
    step("stitch, frame, train", pipeline.rebuild, True)
    db.kv_set("repairs", REPAIRS)


def run() -> None:
    db.init()
    # a database an earlier version already wrote to (interrupted bootstrap, legacy import) still needs repair()
    fresh = _legacy() is None and not db.query_one("SELECT COUNT(*) AS n FROM src_rsam_uwe")["n"]
    catalog.seed()
    step("import legacy database", import_legacy)
    now = int(time.time())
    count = lambda sql: db.query_one(sql)["n"]  # noqa: E731
    if count("SELECT COUNT(*) AS n FROM notices") < 100:
        step("HANS history", hans.backfill)
    if count("SELECT COUNT(*) AS n FROM earthquakes") < 100:
        step("ComCat history", comcat.backfill)
    for name, fn in (("episode table", catalog.fetch), ("USGS status", usgs_status.fetch), ("HANS", hans.fetch), ("HVO messages", hvo_messages.fetch),
                     ("ComCat", comcat.fetch), ("FIRMS", firms.fetch), ("weather", weather.fetch),
                     ("UWD release", tilt_release.fetch_uwd), ("SDH release", tilt_release.fetch_sdh),
                     ("UWD 3-month plot", tilt_plots.fetch_3month), ("UWD 2-day plot", tilt_plots.fetch_2day),
                     ("UWE live", rsam.fetch_uwe), ("UWE.QC live", rsam.fetch_uwe_qc)):
        step(name, fn)
    step("UWE history", rsam.compute_range, "uwe", GRID_START, now)
    step("UWE.QC during legacy outage", rsam.compute_range, "uwe_qc",
         LEGACY_QC_WINDOW[0] - 86400, LEGACY_QC_WINDOW[1] + 86400)
    step("substitute stations for UWE gaps", rsam.fill_substitutes, GRID_START, now)
    step("stitch, frame, train", pipeline.rebuild, True)
    db.kv_set("bootstrap_done", int(time.time()))
    if fresh:
        db.kv_set("repairs", REPAIRS)  # everything above was stored by this version
    else:
        repair()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    t = time.time()
    run()
    log.info("bootstrap finished in %.0f s", time.time() - t)
