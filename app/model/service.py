"""Train, evaluate, and serve the onset-probability models."""
from __future__ import annotations

import json
import logging
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from .. import db, store
from ..analytics import tilt as tilt_an
from ..analytics import tremor as tremor_an
from ..config import DATA_DIR, STORE_AUTOEXPORT
from ..episodes import catalog, suggest
from ..fetchers.base import tracked
from . import features as F
from .hazard import HazardModel
from .ml import MLModel

log = logging.getLogger("model")
HORIZONS = (12, 24, 72)
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
FIRST_TRAIN_EPISODE = 4  # episodes 1–3 were a different (long/continuous) style


def training_frame(now: int | None = None) -> tuple[pd.DataFrame, dict]:
    now = int(now or time.time())
    eps = F.episode_table()
    ep4 = eps[eps["num"] == FIRST_TRAIN_EPISODE].iloc[0]
    t0 = int(np.ceil(ep4["end"] / 3600) * 3600)
    times = np.arange(t0, now - 3600, 3600, dtype=np.int64)
    df = F.build(times, eps)
    df = df[~df["in_episode"] & df["hours_since_end"].notna()].reset_index(drop=True)
    ys = {H: F.labels(df, H, now) for H in (1,) + HORIZONS}
    return df, ys


def _metrics(y: np.ndarray, p: np.ndarray, base_rate: float) -> dict:
    ok = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[ok].astype(int), np.clip(p[ok], 1e-4, 1 - 1e-4)
    if len(y) == 0:
        return {}
    b = brier_score_loss(y, p)
    bc = brier_score_loss(y, np.full(len(y), base_rate))
    out = {"n": int(len(y)), "positives": int(y.sum()), "brier": round(float(b), 4),
           "brier_climatology": round(float(bc), 4),
           "brier_skill": round(float(1 - b / bc), 3) if bc > 0 else None,
           "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 4)}
    if 0 < y.sum() < len(y):
        out["auc"] = round(float(roc_auc_score(y, p)), 3)
    # reliability (5 bins)
    bins = np.clip((p * 5).astype(int), 0, 4)
    out["reliability"] = [
        {"bin": f"{i*20}-{(i+1)*20}%", "n": int((bins == i).sum()),
         "mean_pred": round(float(p[bins == i].mean()), 3) if (bins == i).any() else None,
         "observed": round(float(y[bins == i].mean()), 3) if (bins == i).any() else None}
        for i in range(5)]
    return out


def evaluate(df: pd.DataFrame, ys: dict, n_splits: int = 8) -> dict:
    groups = df["cycle"].to_numpy()
    oof = {m: {H: np.full(len(df), np.nan) for H in HORIZONS} for m in ("hazard", "ml")}
    base = {H: [] for H in HORIZONS}
    for tr, te in GroupKFold(n_splits=n_splits).split(df, groups=groups):
        dtr, dte = df.iloc[tr], df.iloc[te]
        y1 = ys[1][tr]
        ok = ~np.isnan(y1)
        hz = HazardModel().fit(dtr[ok], y1[ok])
        for H, p in hz.predict(dte, HORIZONS).items():
            oof["hazard"][H][te] = p
        mlm = MLModel(HORIZONS).fit(dtr, {H: ys[H][tr] for H in HORIZONS}, groups[tr])
        for H, p in mlm.predict(dte).items():
            oof["ml"][H][te] = p
    res = {}
    open_cycle = df["cycle"].max()
    closed = (df["cycle"] != open_cycle).to_numpy() if np.isnan(df["next_onset"].iloc[-1]) else np.ones(len(df), bool)
    for m in oof:
        res[m] = {"all": {}, "closed": {}}
        for H in HORIZONS:
            y = ys[H]
            res[m]["all"][str(H)] = _metrics(y, oof[m][H], float(np.nanmean(y)))
            yc = np.where(closed, y, np.nan)
            res[m]["closed"][str(H)] = _metrics(yc, oof[m][H], float(np.nanmean(yc)))
    return res


@tracked("model_train")
def train(do_eval: bool = True) -> str:
    t_start = time.time()
    now = int(time.time())
    df, ys = training_frame(now)
    y1 = ys[1]
    ok = ~np.isnan(y1)
    hz = HazardModel().fit(df[ok], y1[ok])
    mlm = MLModel(HORIZONS).fit(df, {H: ys[H] for H in HORIZONS}, df["cycle"].to_numpy())
    metrics = evaluate(df, ys) if do_eval else db.kv_get("model_meta", {}).get("metrics")
    onset_rows = df[ys[1] == 1]
    meta = {
        "trained_at": now,
        "n_rows": int(len(df)),
        "n_onsets": int(np.nansum(ys[1])),
        "cycles": int(df["cycle"].nunique()),
        "train_start": int(df["t"].min()),
        "metrics": metrics,
        "feature_ranges_at_onset": {
            f: [float(onset_rows[f].min()), float(onset_rows[f].max())]
            for f in F.FEATURES if onset_rows[f].notna().any()
        },
        "feature_ranges_all": {
            f: [float(df[f].min()), float(df[f].max())] for f in F.FEATURES if df[f].notna().any()
        },
        "versions": {"hazard": hz.version, "ml": mlm.version},
        "train_seconds": round(time.time() - t_start, 1),
    }
    prev_trained = (db.kv_get("model_meta") or {}).get("trained_at")
    joblib.dump({"hazard": hz, "ml": mlm, "meta": meta}, MODEL_DIR / "current.joblib")
    db.kv_set("model_meta", meta)
    if STORE_AUTOEXPORT:
        try:
            store.register_model(MODEL_DIR / "current.joblib", meta)
        except Exception as e:  # noqa: BLE001 - a store problem must never block serving
            log.warning("model store registration failed: %s", e)
    global _cache
    _cache = None
    rebase_history(prev_trained)
    return f"{meta['n_rows']} rows, {meta['n_onsets']} onsets, {meta['train_seconds']} s"


_cache: dict | None = None


def models() -> dict | None:
    """Load the current model bundle, reloading if another process retrained it."""
    global _cache
    p = MODEL_DIR / "current.joblib"
    if not p.exists():
        return None
    mtime = p.stat().st_mtime
    if _cache is None or _cache.get("_mtime") != mtime:
        _cache = joblib.load(p)
        _cache["_mtime"] = mtime
    return _cache


def _episode_in_progress(row: pd.Series) -> tuple[bool, str | None]:
    open_ev = db.query_one("SELECT label FROM episodes WHERE end_t IS NULL ORDER BY start_t DESC LIMIT 1")
    if open_ev:
        return True, f"catalog lists episode {open_ev['label']} without an end time"
    last_num = db.query_one("SELECT MAX(num) AS n FROM episodes WHERE kind='fountaining'")["n"] or 0
    sug = db.query(
        "SELECT episode_num, phase, sent_unix FROM episode_suggestions WHERE episode_num > ? ORDER BY sent_unix",
        (last_num,))
    started = [s for s in sug if s["phase"] == "onset"]
    ended = [s for s in sug if s["phase"] == "end"]
    if started and (not ended or ended[-1]["sent_unix"] < started[-1]["sent_unix"]):
        return True, f"HVO notice reports episode {started[-1]['episode_num']} began"
    rate = row.get("tilt_rate_6h")
    rr = row.get("rsam_ratio_log")
    if pd.notna(rate) and pd.notna(rr) and rate < -0.8 and rr > np.log10(3):
        return True, "rapid summit deflation with a tremor surge (signal-based detection)"
    return False, None


def _freshness(now: int) -> dict:
    tl = db.query_one("SELECT MAX(t) AS t FROM tilt")["t"]
    rs = db.query_one("SELECT MAX(t) AS t FROM rsam")["t"]
    eq = db.query_one("SELECT last_success AS t FROM source_health WHERE source='comcat'")
    nt = db.query_one("SELECT last_success AS t FROM source_health WHERE source='hans'")
    return {
        "tilt_latest": tl, "tilt_age_min": round((now - tl) / 60, 1) if tl else None,
        "rsam_latest": (rs + 600) if rs else None, "rsam_age_min": round((now - rs - 600) / 60, 1) if rs else None,
        "quakes_checked": eq["t"] if eq else None, "notices_checked": nt["t"] if nt else None,
    }


def _ood(row: pd.Series, meta: dict) -> list[str]:
    notes = []
    rng = meta.get("feature_ranges_at_onset", {})
    rr = row.get("recovery_ratio")
    past = db.query("SELECT label, onset_recovery_ratio AS r FROM episodes WHERE kind='fountaining' "
                    "AND onset_recovery_ratio IS NOT NULL")
    if pd.notna(rr) and past:
        above = [p["label"] for p in past if p["r"] >= rr]
        frac = 1 - len(above) / len(past)
        if frac >= 0.95:
            notes.append(
                f"Tilt has recovered {rr:.2f}× the last episode's deflation, more than at {frac:.0%} of past "
                f"onsets (typical 0.9–1.15×{'; only episode ' + ', '.join(above) + ' was higher' if above else ''}). "
                "The usual tilt threshold hasn't produced an onset this time, so both models are extrapolating.")
    hs = row.get("hours_since_end")
    if pd.notna(hs) and "hours_since_end" in rng and hs > rng["hours_since_end"][1]:
        notes.append(
            f"This pause ({hs/24:.1f} d) is longer than any previous pause "
            f"(max {rng['hours_since_end'][1]/24:.1f} d).")
    since = row.get("last_end")
    if pd.notna(since):
        ev = db.query("SELECT label, notes FROM episodes WHERE kind='non_fountaining' AND start_t > ?", (int(since),))
        for e in ev:
            notes.append(f"Non-fountaining eruptive event since the last episode ({e['label']}): {e['notes']}")
    return notes


def _trend(model: str, p24: float, now: int) -> dict:
    out = {}
    for hrs in (6, 24):
        r = db.query_one("SELECT p24 FROM probability_log WHERE model=? AND t<=? ORDER BY t DESC LIMIT 1",
                         (model, now - hrs * 3600))
        if r and r["p24"] is not None:
            out[f"delta_{hrs}h"] = round(p24 - r["p24"], 4)
    d = out.get("delta_6h", 0.0)
    out["arrow"] = "up" if d > 0.02 else ("down" if d < -0.02 else "flat")
    return out


MAX_INPUT_AGE_S = 45 * 60


def _input_ages(t: int) -> dict[str, float]:
    tl = db.query_one("SELECT MAX(t) AS t FROM tilt")["t"] or 0
    rs = db.query_one("SELECT MAX(t) AS t FROM rsam")["t"] or 0
    return {"tilt": t - tl, "rsam": t - rs}


def _ensure_fresh_inputs(t: int) -> None:
    """Never predict from stale tilt/tremor (e.g. right after the Mac wakes, before the fetch
    jobs have run): those features would come out blank and the number would be meaningless."""
    stale = [k for k, age in _input_ages(t).items() if age > MAX_INPUT_AGE_S]
    if not stale:
        return
    from ..fetchers import fdsn_tremor, usgs_tilt

    if "tilt" in stale:
        usgs_tilt.fetch_fast()
    if "rsam" in stale:
        fdsn_tremor.fetch()
    ages = _input_ages(t)
    still = {k: round(a / 60) for k, a in ages.items() if a > MAX_INPUT_AGE_S}
    if still:
        raise RuntimeError(f"inputs too old to predict (minutes): {still}; keeping the last prediction")


@tracked("model_predict")
def predict_now() -> str:
    m = models()
    if m is None:
        raise RuntimeError("no trained model yet")
    now = int(time.time())
    t = (now // 600) * 600
    _ensure_fresh_inputs(t)
    row_df = F.build(np.array([t]))
    row = row_df.iloc[0]
    meta = m["meta"]
    in_ep, why = _episode_in_progress(row)
    hz, mlm = m["hazard"], m["ml"]
    ph = {H: float(v[0]) for H, v in hz.predict(row_df, HORIZONS).items()}
    pm = {H: float(v[0]) for H, v in mlm.predict(row_df).items()}
    feats = {f: (None if pd.isna(row[f]) else float(row[f])) for f in F.FEATURES}

    def top(contrib: dict, unit: str) -> list[dict]:
        items = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:5]
        return [{"feature": k, "label": F.LABELS.get(k, k), "value": feats.get(k), "effect": round(v, 4),
                 "direction": "raises" if v > 0 else "lowers", "unit": unit} for k, v in items if abs(v) > 1e-6]

    common = {
        "t": t, "in_episode": in_ep, "in_episode_reason": why, "features": feats,
        "last_episode": row["last_label"], "hours_since_last_end": feats["hours_since_end"],
        "warnings": _ood(row, meta), "freshness": _freshness(now),
    }
    out = {}
    for name, p, model, contrib, unit in (
        ("hazard", ph, hz, hz.contributions(row_df), "log-odds of hourly hazard"),
        ("ml", pm, mlm, mlm.contributions(row_df, 24), "Δ P(24 h)"),
    ):
        payload = {**common, "model": name, "version": model.version,
                   "p": {str(H): round(p[H], 4) for H in HORIZONS},
                   "top_factors": top(contrib, unit), "trend": _trend(name, p[24], t)}
        out[name] = payload
        with db.tx() as c:
            c.execute("INSERT OR REPLACE INTO probability_log(t,model,version,p12,p24,p72,payload) VALUES(?,?,?,?,?,?,?)",
                      (t, name, model.version, p[12], p[24], p[72], json.dumps(payload, default=str)))
    return f"hazard p24={ph[24]:.3f}, ml p24={pm[24]:.3f}"


def latest() -> dict:
    out = {}
    for name in ("hazard", "ml"):
        r = db.query_one("SELECT payload FROM probability_log WHERE model=? AND payload NOT LIKE '{\"hindcast\"%' "
                         "ORDER BY t DESC LIMIT 1", (name,))
        if r:
            out[name] = json.loads(r["payload"])
    meta = db.kv_get("model_meta", {})
    return {"primary": "hazard", "models": out, "training": {k: meta.get(k) for k in (
        "trained_at", "n_rows", "n_onsets", "cycles", "train_start", "metrics", "versions")}}


def rebase_history(prev_trained_at: int | None = None) -> int:
    """Redraw the probability history with the current model so the curve is continuous.

    Values the previous model showed live are moved to probability_archive; then every hour of
    the history window is recomputed with the current model. Left of its training time the
    result is in-sample; right of it the model is predicting data it never saw.
    """
    with db.tx() as c:
        c.execute(
            "INSERT OR IGNORE INTO probability_archive(t,model,version,trained_at,p12,p24,p72,payload) "
            "SELECT t, model, version, ?, p12, p24, p72, payload FROM probability_log "
            "WHERE payload NOT LIKE '{\"hindcast\"%'", (prev_trained_at or 0,))
        c.execute("DELETE FROM probability_log")
    n = hindcast()
    predict_now()  # the cards read the latest live row; don't leave them empty until the next run
    return n


def history(hours: int = 168) -> list[dict]:
    """Logged probabilities; `hindcast`=1 marks values recomputed with the current model (in-sample)."""
    t0 = int(time.time()) - hours * 3600
    return db.query(
        "SELECT t, model, p12, p24, p72, (payload LIKE '{\"hindcast\"%') AS hindcast "
        "FROM probability_log WHERE t>=? ORDER BY t", (t0,))


def hindcast(hours: int | None = None, step_s: int = 3600) -> int:
    """Fill probability_log with the current models, back to the end of the series' first
    episode by default (earlier there is no completed cycle to measure from)."""
    m = models()
    if m is None:
        return 0
    now = int(time.time())
    if hours is None:
        first = db.query_one("SELECT MIN(end_t) AS t FROM episodes WHERE kind='fountaining'")["t"]
        t0 = int(first) if first else now - 95 * 86400
    else:
        t0 = now - hours * 3600
    times = np.arange(((t0 // step_s) + 1) * step_s, now - 600, step_s, dtype=np.int64)
    have = {r["t"] for r in db.query("SELECT DISTINCT t FROM probability_log WHERE t>=?", (int(times[0]),))}
    times = np.array([t for t in times if int(t) not in have], dtype=np.int64)
    if not len(times):
        return 0
    df = F.build(times)
    ph = m["hazard"].predict(df, HORIZONS)
    pm = m["ml"].predict(df)
    rows = []
    for i, t in enumerate(times):
        for name, p, ver in (("hazard", ph, m["hazard"].version), ("ml", pm, m["ml"].version)):
            vals = [float(p[H][i]) for H in HORIZONS]
            if bool(df["in_episode"].iloc[i]):
                continue
            rows.append((int(t), name, ver, *vals, json.dumps({"hindcast": True, "p": dict(zip(map(str, HORIZONS), vals))})))
    with db.tx() as c:
        c.executemany("INSERT OR IGNORE INTO probability_log(t,model,version,p12,p24,p72,payload) VALUES(?,?,?,?,?,?,?)", rows)
    return len(rows)
