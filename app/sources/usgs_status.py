"""Alert level / aviation color code from the USGS Volcano API (vhpstatus)."""
from __future__ import annotations

import json

from .. import db
from ..config import VNUM
from .base import get, tracked

URL = "https://volcanoes.usgs.gov/vsc/api/volcanoApi/vhpstatus"


@tracked("usgs_status")
def fetch() -> str:
    r = get(URL)
    r.raise_for_status()
    rows = r.json()
    k = next((x for x in rows if str(x.get("vnum")) == VNUM), None)
    if not k:
        raise ValueError("Kīlauea not present in vhpstatus response")
    db.cache_put("usgs_status", json.dumps(k))
    return f"{k.get('alertLevel')}/{k.get('colorCode')}"


def latest() -> dict | None:
    got = db.cache_get("usgs_status")
    if not got:
        return None
    fetched_at, body = got
    d = json.loads(body)
    return {
        "volcano": d.get("vName"),
        "alert_level": d.get("alertLevel"),
        "color_code": d.get("colorCode"),
        "alert_date_utc": d.get("alertDate"),
        "color_date_utc": d.get("colorDate"),
        "notice_id": d.get("noticeId"),
        "synopsis": d.get("noticeSynopsis"),
        "threat": d.get("nvewsThreat"),
        "volcano_url": d.get("vUrl"),
        "fetched_at": fetched_at,
    }
