"""Summit weather and visibility from Open-Meteo (no key)."""
from __future__ import annotations

import json

from .. import db
from ..config import SUMMIT_LAT, SUMMIT_LON
from .base import get, tracked

URL = "https://api.open-meteo.com/v1/forecast"


@tracked("weather")
def fetch() -> str:
    r = get(URL, params={
        "latitude": SUMMIT_LAT, "longitude": SUMMIT_LON, "elevation": 1100,
        "current": "temperature_2m,relative_humidity_2m,precipitation,cloud_cover,cloud_cover_low,"
                   "wind_speed_10m,wind_direction_10m,visibility,weather_code",
        "hourly": "cloud_cover_low,visibility,precipitation_probability,wind_speed_10m,wind_direction_10m",
        "forecast_days": 2, "timezone": "Pacific/Honolulu", "wind_speed_unit": "kmh",
    })
    r.raise_for_status()
    db.cache_put("weather", r.text)
    cur = r.json().get("current", {})
    return f"{cur.get('temperature_2m')}°C, vis {cur.get('visibility')} m"


def latest() -> dict | None:
    got = db.cache_get("weather")
    if not got:
        return None
    fetched_at, body = got
    d = json.loads(body)
    return {"fetched_at": fetched_at, "current": d.get("current"), "units": d.get("current_units"),
            "hourly": d.get("hourly")}
