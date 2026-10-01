"""Kīlauea fountaining dashboard: FastAPI app + background polling.

    uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import db, scheduler
from .api.routes import router
from .config import DISABLE_SCHEDULER, STATIC_DIR
from .episodes import catalog

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    catalog.seed()
    if not DISABLE_SCHEDULER:
        scheduler.start()
    yield
    scheduler.stop()


app = FastAPI(title="Kīlauea Fountaining Dashboard", version="0.1.0", lifespan=lifespan)
app.include_router(router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def revalidate_static(request: Request, call_next):
    """Make browsers revalidate assets (ETag/Last-Modified) instead of reusing stale copies."""
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def _versioned(match: re.Match) -> str:
    path = STATIC_DIR / match.group(2)
    v = int(path.stat().st_mtime) if path.exists() else 0
    return f'{match.group(1)}/static/{match.group(2)}?v={v}"'


@app.get("/", include_in_schema=False)
def index():
    # Stamp asset URLs with their mtime so an updated CSS/JS is never paired with a cached old one.
    html = (STATIC_DIR / "index.html").read_text()
    html = re.sub(r'((?:src|href)=")/static/([\w./-]+\.(?:css|js))"', _versioned, html)
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})
