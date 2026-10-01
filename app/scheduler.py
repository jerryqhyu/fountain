"""Background polling with APScheduler."""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.background import BackgroundScheduler

log = logging.getLogger("scheduler")

# name -> seconds; used for "stale" flags (stale = 3 missed intervals)
EXPECTED_INTERVAL_S = {
    "usgs_status": 600,
    "hans": 720,
    "comcat": 300,
    "usgs_tilt": 600,
    "usgs_tilt_long": 3600,
    "fdsn_tremor": 600,
    "firms": 1800,
    "weather": 1800,
    "usgs_episode_table": 6 * 3600,
    "tilt_release": 86400,
    "model_predict": 600,
    "model_train": 86400,
    "store_export": 86400,
}

_sched: BackgroundScheduler | None = None


def _next_train_delay() -> int:
    from . import db

    trained = (db.kv_get("model_meta") or {}).get("trained_at")
    if not trained:
        return 24 * 3600  # bootstrap() trains the first model
    return int(max(300, trained + 86400 - time.time()))


def _jobs():
    from . import store
    from .config import STORE_AUTOEXPORT
    from .episodes import catalog
    from .fetchers import comcat, fdsn_tremor, firms, hans, tilt_release, usgs_status, usgs_tilt, weather
    from .model import service

    return [
        # (func, seconds, first-run delay seconds)
        (usgs_status.fetch, 600, 1),
        (hans.fetch, 720, 3),
        (comcat.fetch, 300, 5),
        (usgs_tilt.fetch_fast, 600, 8),
        (usgs_tilt.fetch_slow, 3600, 15),
        (fdsn_tremor.fetch, 600, 20),
        (firms.fetch, 1800, 30),
        (weather.fetch, 1800, 35),
        (catalog.fetch, 6 * 3600, 40),
        (tilt_release.fetch, 86400, 120),
        (service.predict_now, 600, 90),
        # daily, counted from the last actual training so restarts don't keep postponing it
        (service.train, 86400, _next_train_delay()),
    ] + ([(store.export_job, 86400, 900)] if STORE_AUTOEXPORT else [])


def bootstrap() -> None:
    """First-run history backfill + model training, in a thread so the API comes up immediately.

    Every step is idempotent and cheap when the data is already present.
    """
    from . import db
    from .episodes import catalog
    from .fetchers import comcat, fdsn_tremor, hans, tilt_release, usgs_tilt
    from .model import service

    def step(name, fn):
        try:
            log.info("bootstrap: %s", name)
            fn()
        except Exception as e:  # noqa: BLE001
            log.warning("bootstrap step %s failed: %s", name, e)

    def run():
        n = lambda sql: db.query_one(sql)["n"]  # noqa: E731
        if n("SELECT COUNT(*) AS n FROM notices") < 100:
            step("HANS notices", hans.backfill)
        if n("SELECT COUNT(*) AS n FROM earthquakes") < 100:
            step("ComCat", comcat.backfill)
        if n("SELECT COUNT(*) AS n FROM tilt WHERE src LIKE 'release:%'") < 1000:
            step("tilt release", tilt_release.fetch)
        if n("SELECT COUNT(*) AS n FROM episodes") < 40:
            step("episode table", catalog.fetch)
        if n("SELECT COUNT(*) AS n FROM tilt_plot") < 100:
            step("tilt plots", lambda: (usgs_tilt.fetch_slow(), usgs_tilt.fetch_fast()))
        if service.models() is None:
            if n("SELECT COUNT(*) AS n FROM rsam") < 50000:
                # RSAM history takes ~1 h; train a first model without it, retrain afterwards
                step("train (initial)", service.train)
                step("predict", service.predict_now)
                now = int(time.time())
                step("RSAM backfill", lambda: fdsn_tremor.backfill(1733011200, now))
            step("train", service.train)
            step("predict", service.predict_now)

    threading.Thread(target=run, daemon=True, name="bootstrap").start()


def start() -> BackgroundScheduler:
    global _sched
    if _sched:
        return _sched
    # misfire_grace_time=None: a run that was due while the Mac slept runs as soon as it wakes
    # (coalesced to a single run) instead of being skipped until the next interval.
    _sched = BackgroundScheduler(timezone="UTC", job_defaults={"coalesce": True, "max_instances": 1,
                                                                "misfire_grace_time": None})
    now = datetime.now(timezone.utc)
    for fn, every, delay in _jobs():
        name = f"{fn.__module__.split('.')[-1]}.{fn.__name__}"
        _sched.add_job(fn, "interval", seconds=every, next_run_time=now + timedelta(seconds=delay),
                       id=name, name=name, jitter=min(30, every // 20))
    _sched.start()
    bootstrap()
    log.info("scheduler started with %d jobs", len(_sched.get_jobs()))
    return _sched


def stop() -> None:
    global _sched
    if _sched:
        _sched.shutdown(wait=False)
        _sched = None
