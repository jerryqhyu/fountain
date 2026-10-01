"""HVO observatory messages (USGS): "Kilauea Message" posts between Daily Updates.

Scraped from https://www.usgs.gov/observatories/hvo/observatory-messages (newest first).
General "Hawaiian Volcano Observatory Message" entries are kept only if they mention Kīlauea.
Some list entries are truncated by USGS ("…"); the dashboard links to the page.
"""
from __future__ import annotations

import re
from datetime import datetime

from bs4 import BeautifulSoup

from .. import db
from ..config import HST
from .base import get, tracked

URL = "https://www.usgs.gov/observatories/hvo/observatory-messages"
_title = re.compile(r"^(?P<kind>.+?)\s+(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*HST$")


def parse(html: str) -> list[tuple]:
    soup = BeautifulSoup(html, "lxml")
    out = []
    for row in soup.select("div.volcano-message-single"):
        title = row.select_one(".volcano-message-title")
        body = row.select_one(".views-field-value-2 .field-content")
        if not title or not body:
            continue
        m = _title.match(title.get_text(" ", strip=True))
        if not m:
            continue
        kind, text = m["kind"], body.get_text(" ", strip=True)
        if kind != "Kilauea Message" and not re.search(r"k[iī]lauea", text, re.I):
            continue
        t = int(datetime.strptime(m["ts"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=HST).timestamp())
        out.append((f"{kind}|{t}", t, kind, text, text.endswith("...") or text.endswith("…")))
    return out


@tracked("hvo_messages")
def fetch() -> str:
    r = get(URL)
    r.raise_for_status()
    rows = parse(r.text)
    if not rows:
        raise ValueError("no messages found (page layout changed?)")
    db.upsert("messages", ["id", "t", "kind", "text", "truncated"], rows)
    k = sum(1 for x in rows if x[2] == "Kilauea Message")
    return f"{len(rows)} messages on page ({k} Kilauea Messages), newest {rows[0][1]}"


def latest(limit: int = 6) -> list[dict]:
    return db.query("SELECT t, kind, text, truncated FROM messages ORDER BY t DESC LIMIT ?", (limit,))
