"""Shared plumbing for sources: HTTP client, retry, health bookkeeping."""
from __future__ import annotations

import functools
import logging
import time
import traceback
from typing import Callable

import httpx

from .. import db
from ..config import USER_AGENT

log = logging.getLogger("sources")

_client: httpx.Client | None = None


def client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(
            headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
            timeout=httpx.Timeout(60.0, connect=20.0),
            follow_redirects=True,
        )
    return _client


def _backoff(attempt: int, e: Exception) -> None:
    # No network at all (typically the first minute after the Mac wakes): wait it out longer.
    if isinstance(e, httpx.ConnectError):
        time.sleep(min(40, 5 * 2 ** attempt))
    else:
        time.sleep(2 * (attempt + 1))


def get(url: str, *, params: dict | None = None, retries: int = 5, **kw) -> httpx.Response:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = client().get(url, params=params, **kw)
            if r.status_code >= 500 or r.status_code == 429:
                raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
            return r
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            last = e
            if attempt < retries - 1:
                _backoff(attempt, e)
    raise last  # type: ignore[misc]


def post(url: str, *, content: str, headers: dict | None = None, retries: int = 5) -> httpx.Response:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = client().post(url, content=content, headers=headers or {})
            if r.status_code >= 500 or r.status_code == 429:
                raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
            return r
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            last = e
            if attempt < retries - 1:
                _backoff(attempt, e)
    raise last  # type: ignore[misc]


def mark(source: str, ok: bool, error: str | None = None, detail: str | None = None) -> None:
    now = int(time.time())
    with db.tx() as c:
        c.execute("INSERT OR IGNORE INTO source_health(source) VALUES(?)", (source,))
        if ok:
            c.execute(
                "UPDATE source_health SET last_attempt=?, last_success=?, detail=? WHERE source=?",
                (now, now, detail, source),
            )
        else:
            c.execute(
                "UPDATE source_health SET last_attempt=?, last_error=?, last_error_at=? WHERE source=?",
                (now, (error or "")[:2000], now, source),
            )


def tracked(source: str) -> Callable:
    """Wrap a fetch job: never raises, records success/failure in source_health."""

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            try:
                detail = fn(*a, **kw)
                mark(source, True, detail=str(detail) if detail is not None else None)
                return detail
            except Exception as e:  # noqa: BLE001 - jobs must never crash the scheduler
                log.warning("%s failed: %s", source, e)
                log.debug(traceback.format_exc())
                mark(source, False, error=f"{type(e).__name__}: {e}")
                return None

        return wrapper

    return deco
