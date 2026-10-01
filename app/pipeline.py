"""sources → stitched series → feature frame → predictions.

update()  every 5 min: re-stitch, then rebuild frame + predictions from the earliest slot
          whose inputs changed (at least the last 2 days, so rolling features refresh).
rebuild() daily, or when the episode catalog changes: whole grid, retrain, re-infer.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time

from . import db, frame, stitch
from .model import service
from .sources import rsam

log = logging.getLogger("pipeline")
RECENT_S = 2 * 86400


def _catalog_hash() -> str:
    rows = db.query("SELECT label, start_t, end_t FROM episodes ORDER BY start_t")
    return hashlib.sha1(json.dumps(rows).encode()).hexdigest()


def update() -> str:
    now = int(time.time())
    rsam.fill_substitutes(now - 7 * 86400, now)
    if _catalog_hash() != db.kv_get("catalog_hash"):
        return rebuild()  # a new/changed episode shifts features for the whole series
    changed = stitch.run() or {}
    t0 = min([now - RECENT_S] + [t - 86400 for t in changed.values() if t])
    frame.run(t0)
    if service.models() is None:
        service.train()
        t0 = None
    return service.infer(t0) or ""


def rebuild(retrain: bool = True) -> str:
    stitch.run()
    frame.run()
    db.kv_set("catalog_hash", _catalog_hash())
    if retrain or service.models() is None:
        service.train()
    return service.infer() or ""
