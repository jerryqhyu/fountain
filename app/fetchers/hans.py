"""HVO notices for Kīlauea from the USGS HANS public API.

The documented-ish endpoints only return the newest notice per volcano, so we use the
same POST search endpoints the HANS search page uses (search/preflight + search/search,
20 results per page, full HTML included).
"""
from __future__ import annotations

import html as htmllib
import json
import re
import time
from datetime import datetime, timezone

from .. import db
from ..config import VOLCANO_CD
from .base import post, tracked

API = "https://volcanoes.usgs.gov/hans-public/api/"
PAGE = 20
TYPE_TITLES = {
    "DU": "Daily Update", "VAN": "Volcano Activity Notice", "VV": "VONA", "SR": "Status Report",
    "IS": "Information Statement", "WU": "Weekly Update", "MU": "Monthly Update",
}


def html_to_text(h: str) -> str:
    h = re.sub(r"(?i)<br\s*/?>|</p>|</li>|</h\d>", "\n", h or "")
    t = re.sub(r"<[^>]+>", " ", h)
    t = htmllib.unescape(t)
    t = re.sub(r"[ \t\r\f\v]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n\n", t)
    return t.strip()


def _synopsis(text: str) -> str:
    for pat in (r"Summary:\s*(.+?)(?:\n|Volcanic Activity:|Overview:)", r"Activity Summary:\s*(.+?)(?:\n)"):
        m = re.search(pat, text, re.S)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()[:600]
    return re.sub(r"\s+", " ", text)[:300]


def _title(type_cd: str, sent_unix: int) -> str:
    d = datetime.fromtimestamp(sent_unix, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"Kīlauea {TYPE_TITLES.get(type_cd, type_cd)} — {d}"


def _search(page: int, start_unix: int, total: int | None) -> dict:
    body = {
        "obsAbbr": "hvo", "noticeTypeCd": None, "volcCd": VOLCANO_CD, "startDate": None,
        "startUnixtime": start_unix, "endDate": None, "endUnixtime": None, "searchText": None,
        "preflightTotal": total, "pageIndex": page,
    }
    hdr = {"Content-Type": "text/plain;charset=UTF-8"}
    url = API + ("search/search" if total is not None else "search/preflight/")
    r = post(url, content=json.dumps(body), headers=hdr)
    r.raise_for_status()
    return r.json()


def sync(start_unix: int, max_pages: int = 100) -> int:
    pre = _search(0, start_unix, None)
    total = int(pre.get("noticeTotal") or 0)
    if total == 0:
        return 0
    have = {r["notice_id"] for r in db.query("SELECT notice_id FROM notices WHERE sent_unix>=?", (start_unix,))}
    new = 0
    for page in range(min(max_pages, (total + PAGE - 1) // PAGE)):
        data = _search(page, start_unix, total).get("noticeData", [])
        rows = []
        stop = False
        for n in data:
            nid = n.get("noticeIdentifier")
            if nid in have:
                stop = True  # results are newest-first; we've reached known notices
                continue
            text = html_to_text(n.get("noticeHtml", ""))
            sent = int(n.get("sentUnixtime"))
            rows.append((
                nid, sent, n.get("noticeTypeCd"), TYPE_TITLES.get(n.get("noticeTypeCd"), n.get("noticeTypeCd")),
                _title(n.get("noticeTypeCd"), sent), _synopsis(text), text, n.get("noticeHtml"),
                n.get("permLink") or f"https://volcanoes.usgs.gov/hans-public/notice/{nid}",
            ))
        if rows:
            with db.tx() as c:
                c.executemany(
                    "INSERT OR REPLACE INTO notices(notice_id,sent_unix,type_cd,type_title,title,synopsis,text,html,url)"
                    " VALUES(?,?,?,?,?,?,?,?,?)", rows)
            new += len(rows)
        if stop or not data:
            break
        time.sleep(0.3)
    return new


@tracked("hans")
def fetch() -> str:
    last = db.query_one("SELECT MAX(sent_unix) AS t FROM notices")
    start = (last["t"] - 3 * 86400) if last and last["t"] else int(time.time()) - 30 * 86400
    n = sync(start)
    from ..episodes import suggest
    suggest.scan()
    return f"{n} new notices"


def backfill(start_unix: int = 1733000000) -> int:
    return sync(start_unix, max_pages=200)


def latest(limit: int = 5) -> list[dict]:
    rows = db.query(
        "SELECT notice_id, sent_unix, type_cd, type_title, title, synopsis, text, url FROM notices "
        "ORDER BY sent_unix DESC LIMIT ?", (limit,))
    return rows
