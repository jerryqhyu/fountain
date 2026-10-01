"""REST routes."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query

from .. import db
from ..analytics import quakes as quakes_an
from ..analytics import tilt as tilt_an
from ..analytics import tremor as tremor_an
from ..episodes import catalog, suggest
from ..fetchers import hans, usgs_status, weather
from ..model import service as model_service
from ..scheduler import EXPECTED_INTERVAL_S

router = APIRouter(prefix="/api")


def _downsample(s: pd.Series, max_points: int = 3000) -> pd.Series:
    """Thin a long series for display, keeping each bucket's min and max so short, sharp
    features (a 9-hour episode deflation inside a 2-year view) survive."""
    if len(s) <= max_points:
        return s
    n_buckets = max_points // 2
    idx = np.arange(len(s)) * n_buckets // len(s)
    vals = s.to_numpy(float)
    keep = set()
    for b in np.unique(idx):
        pos = np.where(idx == b)[0]
        seg = vals[pos]
        if np.all(np.isnan(seg)):
            continue
        keep.add(int(pos[np.nanargmin(seg)]))
        keep.add(int(pos[np.nanargmax(seg)]))
    return s.iloc[sorted(keep)]


def _fresh(source: str) -> dict:
    h = db.query_one("SELECT * FROM source_health WHERE source=?", (source,)) or {}
    now = int(time.time())
    exp = EXPECTED_INTERVAL_S.get(source)
    last = h.get("last_success")
    return {
        "last_success": last,
        "age_s": (now - last) if last else None,
        "stale": (last is None) or (exp is not None and now - last > 3 * exp),
        "last_error": h.get("last_error") if (h.get("last_error_at") or 0) > (last or 0) else None,
    }


@router.get("/status")
def status():
    s = usgs_status.latest()
    return {"status": s, "freshness": _fresh("usgs_status")}


@router.get("/notices")
def notices(limit: int = Query(5, ge=1, le=50)):
    return {"notices": hans.latest(limit), "freshness": _fresh("hans"),
            "hans_url": "https://volcanoes.usgs.gov/hans-public/search/"}


def _episode_markers(t0: int, t1: int) -> list[dict]:
    return [
        {"label": e["label"], "kind": e["kind"], "start": e["start_t"], "end": e["end_t"]}
        for e in catalog.catalog() if (e["end_t"] or t1) >= t0 and e["start_t"] <= t1
    ]


@router.get("/tilt")
def tilt(hours: float = Query(72, gt=0, le=24 * 800)):
    now = int(time.time())
    t0 = now - int(hours * 3600)
    s = tilt_an.load(t0 - 48 * 3600)
    rate = tilt_an.rate_series(s, 3.0)
    s_w, rate_w = s[s.index >= t0], rate[rate.index >= t0]
    src = db.query("SELECT src, MIN(t) t0, MAX(t) t1, COUNT(*) n FROM tilt WHERE t>=? GROUP BY src ORDER BY t0", (t0,))
    state = tilt_an.current_state()
    # inflation since the last fountaining episode ended, vs. historical onset levels
    infl = None
    last = catalog.last_fountaining()
    if last and last["end_t"]:
        full = tilt_an.load(last["start_t"] - 3600)
        tr = tilt_an.trough(full, last["start_t"], last["end_t"])
        after = full[full.index >= last["end_t"]]
        if tr is not None and len(after):
            a = _downsample(after - tr, 1500)
            defl = state.get("last_deflation_urad")
            thr = tilt_an.onset_thresholds(15)
            infl = {
                "since": last["end_t"], "episode": last["label"], "trough": tr,
                "t": a.index.tolist(), "v": [round(x, 3) for x in a.tolist()],
                "last_deflation_urad": defl, "thresholds": thr,
                "ratio_thresholds_urad": ({k: v * defl for k, v in thr["recovery_ratio"].items()}
                                          if defl and thr.get("recovery_ratio") else None),
            }
    d = _downsample(s_w)
    rd = _downsample(rate_w)
    return {
        "station": "UWD", "component": "azimuth 300° (radial)", "units": "µrad",
        "t": d.index.tolist(), "v": [round(x, 3) for x in d.tolist()],
        "rate_t": rd.index.tolist(), "rate": [round(x, 4) for x in rd.tolist()], "rate_units": "µrad/h (3 h fit)",
        "episodes": _episode_markers(t0, now), "state": state, "inflation": infl,
        "sources": src, "stitch": db.kv_get("tilt_stitch"),
        "freshness": {"usgs_tilt": _fresh("usgs_tilt"), "tilt_release": _fresh("tilt_release")},
        "note": "Live values are digitized from HVO's UWD tilt plots and aligned to the USGS 1-minute "
                "data release; absolute level is arbitrary, changes are meaningful.",
    }


@router.get("/tremor")
def tremor(hours: float = Query(72, gt=0, le=24 * 800)):
    now = int(time.time())
    t0 = now - int(hours * 3600)
    s = tremor_an.load(t0)
    d = _downsample(s)
    return {
        "station": "HV.UWE..HHZ", "band_hz": [1, 5], "window_s": 600, "units": "µm/s (mean |v|)",
        "t": d.index.tolist(), "v": [round(x, 5) for x in d.tolist()],
        "episodes": _episode_markers(t0, now), "current": tremor_an.current(),
        "freshness": _fresh("fdsn_tremor"),
    }


@router.get("/earthquakes")
def earthquakes(days: float = Query(7, gt=0, le=800)):
    out = quakes_an.summary(int(np.ceil(days)))
    out["freshness"] = _fresh("comcat")
    return out


@router.get("/episodes")
def episodes():
    rows = catalog.catalog()
    return {"episodes": rows, "suggestions": suggest.pending(), "thresholds": tilt_an.onset_thresholds(15),
            "freshness": _fresh("usgs_episode_table"),
            "source_url": "https://www.usgs.gov/volcanoes/kilauea/science/eruption-information"}


@router.get("/probability")
def probability():
    out = model_service.latest()
    out["freshness"] = {"predict": _fresh("model_predict"), "train": _fresh("model_train")}
    out["disclaimer"] = ("Experimental, unofficial statistical estimate built from public data. It is not a "
                         "USGS forecast. Follow HVO for hazard information.")
    return out


def _series_start() -> int | None:
    row = db.query_one("SELECT MIN(start_t) AS t FROM episodes")
    return row["t"] if row else None


@router.get("/probability/history")
def probability_history(hours: float = Query(168, gt=0, le=24 * 800)):
    meta = db.kv_get("model_meta", {}) or {}
    rows = model_service.history(int(hours))
    out = []
    for model in ("hazard", "ml"):
        r = [x for x in rows if x["model"] == model]
        if len(r) > 3000:  # thin long views, keeping each bucket's lowest and highest p24
            s = pd.Series([x["p24"] for x in r])
            r = [r[i] for i in _downsample(s, 3000).index]
        out += r
    return {"history": out, "trained_at": meta.get("trained_at"), "series_start": _series_start()}


@router.get("/firms")
def firms(days: float = Query(7, gt=0, le=60)):
    t0 = int(time.time() - days * 86400)
    rows = db.query("SELECT t, lat, lon, frp, bright, satellite, confidence, source FROM firms WHERE t>=? ORDER BY t", (t0,))
    return {"detections": rows, "freshness": _fresh("firms")}


@router.get("/weather")
def weather_route():
    return {"weather": weather.latest(), "freshness": _fresh("weather")}


@router.get("/health")
def health():
    rows = db.query("SELECT * FROM source_health ORDER BY source")
    now = int(time.time())
    for r in rows:
        exp = EXPECTED_INTERVAL_S.get(r["source"])
        r["expected_interval_s"] = exp
        r["age_s"] = now - r["last_success"] if r["last_success"] else None
        r["stale"] = r["last_success"] is None or (exp is not None and now - r["last_success"] > 3 * exp)
    return {"now": now, "sources": rows}
