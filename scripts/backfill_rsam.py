"""Backfill 10-minute RSAM from Dec 2024 to now. Safe to re-run; skips filled chunks.

    uv run python -m scripts.backfill_rsam [--since 2024-12-01]
"""
from __future__ import annotations

import argparse
import logging
import time

from obspy import UTCDateTime

from app import db
from app.fetchers import fdsn_tremor

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2024-12-01")
    ap.add_argument("--until", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    db.init()
    start = int(UTCDateTime(args.since).timestamp)
    end = int(UTCDateTime(args.until).timestamp) if args.until else int(time.time())
    # Newest first so the live window is useful immediately, then walk back.
    fdsn_tremor.backfill(end - 7 * 86400, end)
    fdsn_tremor.backfill(start, end - 7 * 86400)
    logging.info("done")
