"""Background jobs: one per source at its own cadence, plus the pipeline."""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.background import BackgroundScheduler

from . import db

log = logging.getLogger("scheduler")

# source_health key -> expected interval (s); "stale" = 3 missed intervals
EXPECTED_INTERVAL_S = {
    "usgs_status": 600, "hans": 720, "hvo_messages": 600, "usgs_episode_table": 6 * 3600, "comcat": 300,
    "firms": 1800, "weather": 1800,
    "uwd_release": 86400, "sdh_release": 86400, "uwd_plot2d": 600, "uwd_plot3m": 3600,
    "uwe": 600, "uwe_qc": 600,
    "stitch": 300, "frame": 300, "infer": 300, "train": 86400,
}

_sched: BackgroundScheduler | None = None


def _next_rebuild_delay() -> int:
    trained = (db.kv_get("model_meta") or {}).get("trained_at")
    return int(max(300, trained + 86400 - time.time())) if trained else 86400


def _jobs():
    from . import pipeline
    from .episodes import catalog
    from .sources import comcat, firms, hans, hvo_messages, rsam, tilt_plots, tilt_release, usgs_status, weather

    return [
        # (func, every s, first-run delay s)
        (usgs_status.fetch, 600, 1), (hans.fetch, 720, 3), (hvo_messages.fetch, 600, 4), (comcat.fetch, 300, 5),
        (tilt_plots.fetch_2day, 600, 8), (tilt_plots.fetch_3month, 3600, 12),
        (rsam.fetch_uwe, 600, 15), (rsam.fetch_uwe_qc, 600, 25),
        (firms.fetch, 1800, 30), (weather.fetch, 1800, 35), (catalog.fetch, 6 * 3600, 40),
        (tilt_release.fetch_uwd, 86400, 120), (tilt_release.fetch_sdh, 86400, 150),
        (pipeline.update, 300, 90),
        (pipeline.rebuild, 86400, _next_rebuild_delay()),  # daily, counted from the last training
    ]


def start() -> BackgroundScheduler:
    global _sched
    if _sched:
        return _sched
    # misfire_grace_time=None: a run due while the Mac slept runs once on wake instead of being skipped
    _sched = BackgroundScheduler(timezone="UTC", job_defaults={"coalesce": True, "max_instances": 1,
                                                                "misfire_grace_time": None})
    now = datetime.now(timezone.utc)
    for fn, every, delay in _jobs():
        name = f"{fn.__module__.split('.')[-1]}.{fn.__name__}"
        _sched.add_job(fn, "interval", seconds=every, next_run_time=now + timedelta(seconds=delay),
                       id=name, name=name, jitter=min(30, every // 20))
    _sched.start()
    if not db.query_one("SELECT 1 AS x FROM src_tilt_uwd_release LIMIT 1"):
        from . import bootstrap
        threading.Thread(target=bootstrap.run, daemon=True, name="bootstrap").start()
    log.info("scheduler started with %d jobs", len(_sched.get_jobs()))
    return _sched


def stop() -> None:
    global _sched
    if _sched:
        _sched.shutdown(wait=False)
        _sched = None
