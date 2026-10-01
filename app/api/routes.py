"""REST routes."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query

from .. import db, frame
from ..config import GRID_S, HST
from ..episodes import catalog, suggest
from ..model import service
from ..scheduler import EXPECTED_INTERVAL_S
from ..sources import HEALTH_GROUPS, SUBSTITUTE_TREMOR, TILT_SOURCES, TREMOR_SOURCES, hans, hvo_messages, usgs_status, weather

router = APIRouter(prefix="/api")
MAX_POINTS = 3000


# ------------------------------------------------------------------ helpers
def thin(t: np.ndarray, key: np.ndarray, max_points: int = MAX_POINTS) -> np.ndarray:
    """Indices to keep for display: each bucket's min and max of `key`, so short features survive."""
    n = len(t)
    if n <= max_points:
        return np.arange(n)
    buckets = np.arange(n) * (max_points // 2) // n
    keep = []
    for b in np.unique(buckets):
        pos = np.where(buckets == b)[0]
        seg = key[pos]
        if np.all(np.isnan(seg)):
            keep.append(pos[0])
            continue
        keep += [pos[np.nanargmin(seg)], pos[np.nanargmax(seg)]]
    return np.unique(keep)


def rows_since(table: str, cols: str, t0: int) -> pd.DataFrame:
    names = [c.strip() for c in cols.split(",")]
    return pd.DataFrame(db.query(f"SELECT {cols} FROM {table} WHERE t>=? ORDER BY t", (t0,)), columns=names)


def series_json(df: pd.DataFrame, value: str = "v", digits: int = 4) -> dict:
    if df.empty:
        return {"t": [], value: []}
    idx = thin(df["t"].to_numpy(), df[value].to_numpy(float))
    d = df.iloc[idx]
    out = {"t": d["t"].astype(int).tolist(), value: [None if pd.isna(x) else round(float(x), digits) for x in d[value]]}
    if "src" in d:
        out["src"] = d["src"].tolist()
    return out


def health_row(key: str) -> dict:
    h = db.query_one("SELECT * FROM source_health WHERE source=?", (key,)) or {}
    now = int(time.time())
    exp = EXPECTED_INTERVAL_S.get(key)
    last = h.get("last_success")
    err = (h.get("last_error_at") or 0) > (last or 0)
    return {"key": key, "last_success": last, "age_s": now - last if last else None,
            "stale": bool(last and exp and now - last > 3 * exp), "error": h.get("last_error") if err else None,
            "detail": h.get("detail")}


def episode_marks(t0: int) -> list[dict]:
    return [{"label": e["label"], "kind": e["kind"], "start": e["start_t"], "end": e["end_t"]}
            for e in catalog.catalog() if (e["end_t"] or 2**62) >= t0]


# ------------------------------------------------------------------ official / context
@router.get("/status")
def status():
    return {"status": usgs_status.latest(), "freshness": health_row("usgs_status")}


@router.get("/notices")
def notices(limit: int = Query(5, ge=1, le=50)):
    return {"notices": hans.latest(limit), "freshness": health_row("hans"),
            "hans_url": "https://volcanoes.usgs.gov/hans-public/search/"}


@router.get("/messages")
def messages(limit: int = Query(6, ge=1, le=50)):
    return {"messages": hvo_messages.latest(limit), "freshness": health_row("hvo_messages"),
            "url": hvo_messages.URL}


@router.get("/episodes")
def episodes():
    return {"episodes": catalog.catalog(), "suggestions": suggest.pending(), "freshness": health_row("usgs_episode_table"),
            "source_url": "https://www.usgs.gov/volcanoes/kilauea/science/eruption-information"}


@router.get("/firms")
def firms(days: float = Query(7, gt=0, le=60)):
    rows = db.query("SELECT t, lat, lon, frp, bright, satellite, confidence, source FROM firms WHERE t>=? ORDER BY t",
                    (int(time.time() - days * 86400),))
    return {"detections": rows, "freshness": health_row("firms")}


@router.get("/weather")
def weather_route():
    return {"weather": weather.latest(), "freshness": health_row("weather")}


@router.get("/health")
def health():
    return {"now": int(time.time()), "groups": [
        {"name": name, "sources": [{**health_row(k), "label": label} for k, label in items]}
        for name, items in HEALTH_GROUPS]}


# ------------------------------------------------------------------ series: every source + the stitched result
@router.get("/series/tilt")
def tilt(hours: float = Query(72, gt=0, le=24 * 800)):
    t0 = int(time.time() - hours * 3600)
    info = db.kv_get("stitch:tilt", {}) or {}
    offsets = info.get("offsets", {})
    stitched = rows_since("series_tilt", "t, v, src", t0)
    sources = []
    for s in TILT_SOURCES:
        if s.key == "sdh_release":  # SDH is only meaningful through its per-gap fit: show the fitted slots
            d = stitched[stitched["src"] == "sdh_release"][["t", "v"]]
            mapping = {"fits": [f for f in info.get("sdh_fits", []) if f["to"] >= t0]}
        else:
            d = rows_since(s.table, "t, v", t0)
            off = offsets.get(s.key) if s.key != "uwd_release" else 0.0
            mapping = {"offset": off}
            if off is None:
                d = d.iloc[0:0]
            else:
                d = d.assign(v=d["v"] + off)
        sources.append({"key": s.key, "label": s.label, "note": s.note, "mapping": mapping,
                        **series_json(d), "health": health_row(s.key)})
    full = frame.series("series_tilt", np.arange(t0 - 2 * 86400 - t0 % GRID_S, int(time.time()) + 1, GRID_S))
    rate = frame.rate(full, 3).loc[t0:]
    rate_df = pd.DataFrame({"t": rate.index, "v": rate.to_numpy()})
    return {"units": "µrad", "component": "UWD azimuth 300° (radial)", "stitched": series_json(stitched),
            "sources": sources, "rate": series_json(rate_df.dropna(), digits=5),
            "slots_by_source": info.get("slots_by_source"), "missing_slots": info.get("missing"),
            "episodes": episode_marks(t0), "inflation": inflation_since_last()}


@router.get("/series/tremor")
def tremor(hours: float = Query(72, gt=0, le=24 * 800)):
    t0 = int(time.time() - hours * 3600)
    info = db.kv_get("stitch:tremor", {}) or {}
    stitched = rows_since("series_rsam", "t, v, src", t0)
    sources = []
    for s in TREMOR_SOURCES:
        d = rows_since(s.table, "t, v", t0 - 86400)
        mapping: dict = {"scale": 1.0}
        if s.key in SUBSTITUTE_TREMOR:  # show substitutes scaled the way they were used, around their gaps
            parts, used = [], []
            for r in info.get("ratios", []):
                if r["station"] == s.key and r["to"] >= t0 - 86400:
                    w = d[(d["t"] >= r["from"] - 86400) & (d["t"] <= r["to"] + 86400)]
                    parts.append(w.assign(v=w["v"] * r["ratio"]))
                    used.append(r)
            d = pd.concat(parts) if parts else d.iloc[0:0]
            mapping = {"ratios": used}
        d = d[d["t"] >= t0]
        sources.append({"key": s.key, "label": s.label, "note": s.note, "mapping": mapping,
                        **series_json(d, digits=5), "health": health_row(s.key)})
    return {"units": "µm/s", "band_hz": [1, 5], "window_s": 600, "stitched": series_json(stitched, digits=5),
            "sources": sources, "slots_by_source": info.get("slots_by_source"), "missing_slots": info.get("missing"),
            "episodes": episode_marks(t0)}


def inflation_since_last() -> dict | None:
    eps = [e for e in catalog.fountaining() if e["end_t"]]
    if not eps:
        return None
    last = eps[-1]
    d = rows_since("series_tilt", "t, v", last["start_t"])
    if d.empty:
        return None
    during = d[(d["t"] >= last["start_t"]) & (d["t"] <= last["end_t"] + 7200)]
    if during.empty:
        return None
    trough = float(during["v"].min())
    after = d[d["t"] >= last["end_t"]].assign(v=lambda x: x["v"] - trough)
    defl = db.query_one("SELECT deflation_urad AS d FROM episodes WHERE label=?", (last["label"],))["d"]
    recent = db.query("SELECT onset_recovery_urad AS rec, onset_recovery_ratio AS ratio FROM episodes "
                      "WHERE kind='fountaining' AND onset_recovery_ratio IS NOT NULL ORDER BY start_t")[-15:]
    thr = None
    if recent:
        rec = np.array([r["rec"] for r in recent], float)
        ratio = np.array([r["ratio"] for r in recent], float)
        thr = {"n": len(recent),
               "recovery_urad": {q: float(np.percentile(rec, p)) for q, p in (("p10", 10), ("median", 50), ("p90", 90))},
               "recovery_ratio": {q: float(np.percentile(ratio, p)) for q, p in (("p10", 10), ("median", 50), ("p90", 90))}}
    return {"episode": last["label"], "since": last["end_t"], "trough": trough, "last_deflation_urad": defl,
            **series_json(after), "thresholds": thr,
            "ratio_thresholds_urad": ({k: v * defl for k, v in thr["recovery_ratio"].items()} if thr and defl else None)}


@router.get("/earthquakes")
def earthquakes(days: float = Query(7, gt=0, le=800)):
    t0 = int(time.time() - days * 86400)
    rows = db.query("SELECT id, t, lat, lon, depth, mag, place, region FROM earthquakes WHERE t>=? ORDER BY t", (t0,))
    df = pd.DataFrame(rows, columns=["id", "t", "lat", "lon", "depth", "mag", "place", "region"])
    depth = pd.cut(df["depth"], [-10, 5, 20, 1000], labels=["shallow (<5 km)", "intermediate (5–20 km)", "deep (>20 km)"])
    last24 = df[df["t"] >= time.time() - 86400]
    counts = {"total": int(len(df)), "last_24h": int(len(last24)), "summit_last_24h": int((last24["region"] == "summit").sum()),
              "by_region": df["region"].value_counts().to_dict(), "by_depth": depth.value_counts().to_dict(),
              "max_mag": float(df["mag"].max()) if df["mag"].notna().any() else None}
    return {"events": df.replace({np.nan: None}).to_dict("records"), "counts": counts, "freshness": health_row("comcat")}


# ------------------------------------------------------------------ frame and predictions
@router.get("/frame")
def frame_rows(hours: float = Query(72, gt=0, le=24 * 800)):
    t0 = int(time.time() - hours * 3600)
    df = frame.load(t0)
    if df.empty:
        return {"t": [], "columns": {}}
    keep = thin(df.index.to_numpy(), df["recovery_ratio"].astype(float).to_numpy())
    d = df.iloc[keep]
    cols = {c: [None if pd.isna(x) else (x if isinstance(x, str) else round(float(x), 5)) for x in d[c]]
            for c in frame.FEATURES + ["in_episode", "tilt_value", "rsam_1h_ums"]}
    return {"t": d.index.astype(int).tolist(), "columns": cols, "labels": frame.LABELS,
            "required": {m: r for m, r in service.REQUIRED.items()}}


@router.get("/predictions")
def predictions(hours: float = Query(168, gt=0, le=24 * 800)):
    t0 = int(time.time() - hours * 3600)
    out = {}
    for name in service.PRED_TABLE:
        p = service.predictions(name, t0)
        if len(p):
            p = p.iloc[thin(p["t"].to_numpy(), p["p24"].astype(float).to_numpy())]
        out[name] = {"t": p["t"].astype(int).tolist(),
                     **{c: [None if pd.isna(x) else round(float(x), 5) for x in p[c]] for c in ("p12", "p24", "p72")}}
    meta = db.kv_get("model_meta", {}) or {}
    return {"models": out, "trained_at": meta.get("trained_at")}


@router.get("/probability")
def probability():
    out = service.latest()
    out["freshness"] = {k: health_row(k) for k in ("stitch", "frame", "infer", "train")}
    out["disclaimer"] = ("Experimental, unofficial statistical estimate built from public data. It is not a USGS "
                         "forecast. Follow HVO for hazard information.")
    return out
