"""Walk-forward training of the hazard model, and its predictions on the grid.

The model is refitted once per calendar month (Hawaii time). The fit for a month uses only
frame rows whose outcome was known before that month began, and predicts only that month, so
every stored probability is out-of-sample. The first fit covers December 2024 – June 2025;
there are no predictions before July 2025.

pred_hazard(t, p12) has one row per hourly slot from July 2025. p12 is NULL when an episode is
under way or when any input the model requires is genuinely missing (no imputation).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .. import db, frame
from ..config import DATA_DIR, HST
from ..frame import FEATURES, LABELS
from ..sources.base import tracked
from ..stitch import runs
from .hazard import GROUPS, HazardModel

log = logging.getLogger("model")
HORIZON_H = 12
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "walkforward.joblib"
FIRST_TRAIN_EPISODE = 4  # episodes 1–3 were long/continuous, a different style
FIRST_PREDICTED_MONTH = (2025, 7)  # warm-up: the first fit sees December 2024 – June 2025
REQUIRED = list(GROUPS)
GRACE_SLOTS = 1  # the newest slot may still be waiting for its inputs: live sources lag 10–40 min


# ------------------------------------------------------------------ training
def month_starts(now: int) -> list[int]:
    """Start (HST) of every calendar month from the first predicted one up to the current one."""
    out, (y, m) = [], FIRST_PREDICTED_MONTH
    while (t := int(datetime(y, m, 1, tzinfo=HST).timestamp())) <= now:
        out.append(t)
        y, m = y + m // 12, m % 12 + 1
    return out


def labels(df: pd.DataFrame, horizon_h: float, now: int) -> np.ndarray:
    """1 if the next onset is within (t, t+H]; 0 if not; NaN if not yet knowable at `now`."""
    h = horizon_h * 3600
    t = df.index.to_numpy(float)
    nxt = df["next_onset"].to_numpy(float)
    y = np.where(nxt - t <= h, 1.0, 0.0)  # NaN next_onset compares False
    y[t + h > now] = np.nan
    y[(nxt <= now) & (nxt - t <= h)] = 1.0  # an onset that has already happened is known early
    return y


def valid_mask(df: pd.DataFrame) -> np.ndarray:
    return ((df["in_episode"] == 0) & df[REQUIRED].notna().all(axis=1)).to_numpy()


def fit_month(df: pd.DataFrame, cutoff: int) -> HazardModel | None:
    """Model for the month starting at `cutoff`, from the pause-hours whose outcome was known by then."""
    ep4 = db.query_one("SELECT end_t FROM episodes WHERE num=?", (FIRST_TRAIN_EPISODE,))
    y1 = labels(df, 1, cutoff)
    rows = valid_mask(df) & ~np.isnan(y1) & (df.index.to_numpy() > ((ep4 or {}).get("end_t") or 0))
    if y1[rows].sum() < 2:
        return None
    hz = HazardModel().fit(df[rows], y1[rows])
    onsets = df[rows][y1[rows] == 1]
    hz.fit_info_ = {
        "cutoff": int(cutoff), "n_rows": int(rows.sum()), "n_onsets": int(y1[rows].sum()),
        "cycles": int(df[rows]["last_label"].nunique()), "train_start": int(df.index[rows].min()),
        "feature_ranges_at_onset": {f: [float(onsets[f].min()), float(onsets[f].max())]
                                    for f in FEATURES if onsets[f].notna().any()},
    }
    return hz


def predict(fits: list[HazardModel], df: pd.DataFrame) -> np.ndarray:
    """P(onset within 12 h) per row, each month from the model fitted before it began."""
    p = np.full(len(df), np.nan)
    t = df.index.to_numpy()
    ok = valid_mask(df)
    for i, hz in enumerate(fits):
        # the newest model also covers later rows until its successor is fitted
        end = fits[i + 1].fit_info_["cutoff"] if i + 1 < len(fits) else 2**62
        sel = np.where(ok & (t >= hz.fit_info_["cutoff"]) & (t < end))[0]
        if len(sel):
            p[sel] = hz.predict(df.iloc[sel], (HORIZON_H,))[HORIZON_H]
    return p


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    ok = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[ok].astype(int), np.clip(p[ok], 1e-4, 1 - 1e-4)
    if not len(y):
        return {}
    base = float(y.mean())
    b, bc = brier_score_loss(y, p), brier_score_loss(y, np.full(len(y), base))
    out = {"n": int(len(y)), "positives": int(y.sum()), "base_rate": round(base, 4), "brier": round(float(b), 4),
           "brier_climatology": round(float(bc), 4),
           "brier_skill": round(float(1 - b / bc), 3) if bc > 0 else None,
           "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 4)}
    if 0 < y.sum() < len(y):
        out["auc"] = round(float(roc_auc_score(y, p)), 3)
    return out


def evaluate(df: pd.DataFrame, p: np.ndarray, now: int) -> dict:
    """Skill of the stored (out-of-sample) predictions, for completed cycles and for all rows."""
    y = labels(df, HORIZON_H, now)
    open_pause = np.isnan(df["next_onset"].to_numpy(float))
    return {"all": _metrics(y, p), "closed": _metrics(np.where(open_pause, np.nan, y), p)}


def _write(df: pd.DataFrame, p: np.ndarray) -> None:
    db.upsert("pred_hazard", ["t", "p12"],
              [(int(t), None if np.isnan(v) else float(v)) for t, v in zip(df.index, p)])


@tracked("train")
def train() -> str:
    """Refit every month's model and recompute all predictions."""
    t_start = time.time()
    now = int(t_start)
    df = frame.load()
    df[FEATURES] = df[FEATURES].astype(float)
    fits = [hz for c in month_starts(now) if (hz := fit_month(df, c)) is not None]
    if not fits:
        raise RuntimeError("not enough history to fit the first month's model yet")
    out = df[df.index >= fits[0].fit_info_["cutoff"]]
    p = predict(fits, out)
    last = fits[-1].fit_info_
    meta = {
        "trained_at": now, "version": HazardModel.version, "horizon_h": HORIZON_H, "required": REQUIRED,
        **{k: last[k] for k in ("n_rows", "n_onsets", "cycles", "train_start", "feature_ranges_at_onset")},
        "fit_through": last["cutoff"], "first_prediction": fits[0].fit_info_["cutoff"],
        "refits": [hz.fit_info_["cutoff"] for hz in fits],
        "metrics": evaluate(out, p, now),
        "train_seconds": round(time.time() - t_start, 1),
    }
    joblib.dump({"fits": fits, "meta": meta}, MODEL_PATH)
    with db.tx() as c:
        c.execute("DELETE FROM pred_hazard")
    _write(out, p)
    db.kv_set("model_meta", meta)
    return (f"{len(fits)} monthly fits; latest on {last['n_rows']} hourly rows, {last['n_onsets']} onsets "
            f"through {last['cutoff']}; {meta['train_seconds']} s")


_cache: dict | None = None


def models() -> dict | None:
    """Current model bundle, reloaded if another process retrained it."""
    global _cache
    if not MODEL_PATH.exists():
        return None
    mtime = MODEL_PATH.stat().st_mtime
    if _cache is None or _cache.get("_mtime") != mtime:
        _cache = joblib.load(MODEL_PATH)
        _cache["_mtime"] = mtime
    return _cache


def stale(now: int | None = None) -> bool:
    """True when there is no model yet, or a new month has begun since the last fit."""
    m = models()
    months = month_starts(int(now or time.time()))
    return m is None or (bool(months) and months[-1] > m["meta"]["fit_through"])


# ------------------------------------------------------------------ inference
@tracked("infer")
def infer(t0: int | None = None) -> str:
    m = models()
    if m is None:
        raise RuntimeError("no trained model yet")
    df = frame.load(max(t0 or 0, m["meta"]["first_prediction"]))
    if df.empty:
        return "no frame rows"
    df[FEATURES] = df[FEATURES].astype(float)
    p = predict(m["fits"], df)
    _write(df, p)
    return f"{int((~np.isnan(p)).sum())}/{len(df)} slots valid"


def predictions(t0: int, t1: int | None = None) -> pd.DataFrame:
    rows = db.query("SELECT t, p12 FROM pred_hazard WHERE t>=? AND t<=? ORDER BY t", (t0, t1 or 2**62))
    return pd.DataFrame(rows, columns=["t", "p12"])


def missing_spans(t0: int, t1: int | None = None) -> list[dict]:
    """Runs of pause hours in [t0, t1] with no prediction because a required input is missing."""
    first = (db.kv_get("model_meta", {}) or {}).get("first_prediction")
    if first is None:
        return []
    df = frame.load(max(t0, first), t1)
    if df.empty:
        return []
    t = df.index.to_numpy()
    gone = df[REQUIRED].astype(float).isna()
    out = []
    for i0, i1 in runs(((df["in_episode"] == 0) & gone.any(axis=1)).to_numpy()):
        feats = [f for f in REQUIRED if gone[f].iloc[i0:i1 + 1].any()]
        out.append({"from": int(t[i0]), "to": int(t[i1]), "features": feats, "labels": [LABELS.get(f, f) for f in feats]})
    return out


# ------------------------------------------------------------------ latest, for the card
def current_slot(fr: pd.DataFrame, pr: pd.DataFrame) -> tuple[int, bool]:
    """(slot the card reports, whether it has a prediction).

    The newest slot with a prediction, looking back only GRACE_SLOTS: beyond that an absent
    prediction is reported as absent instead of being covered by an older value.
    """
    recent = fr.index[-(GRACE_SLOTS + 1):]
    have = pr[pr["t"].isin(recent)]
    if len(have):
        return int(have["t"].iloc[-1]), True
    return int(recent[0]), False


def _episode_in_progress(row: pd.Series) -> tuple[bool, str | None]:
    open_ev = db.query_one("SELECT label FROM episodes WHERE end_t IS NULL ORDER BY start_t DESC LIMIT 1")
    if open_ev:
        return True, f"catalog lists episode {open_ev['label']} without an end time"
    if pd.notna(row.get("tilt_rate_6h")) and pd.notna(row.get("rsam_ratio_log")) \
            and row["tilt_rate_6h"] < -0.8 and row["rsam_ratio_log"] > np.log10(3):
        return True, "rapid summit deflation with a tremor surge (signal-based detection)"
    return False, None


def _warnings(row: pd.Series, meta: dict) -> list[str]:
    notes = []
    rr = row.get("recovery_ratio")
    past = db.query("SELECT label, onset_recovery_ratio AS r FROM episodes WHERE kind='fountaining' "
                    "AND onset_recovery_ratio IS NOT NULL")
    if pd.notna(rr) and past:
        above = [p["label"] for p in past if p["r"] >= rr]
        frac = 1 - len(above) / len(past)
        if frac >= 0.95:
            notes.append(
                f"Tilt has recovered {rr:.2f}× the last episode's deflation, more than at {frac:.0%} of past onsets "
                f"(typical 0.9–1.15×{'; only episode ' + ', '.join(above) + ' was higher' if above else ''}). "
                "The usual tilt threshold hasn't produced an onset this time, so the model is extrapolating.")
    rng = meta.get("feature_ranges_at_onset", {})
    hs = row.get("hours_since_end")
    if pd.notna(hs) and "hours_since_end" in rng and hs > rng["hours_since_end"][1]:
        notes.append(f"This pause ({hs / 24:.1f} d) is longer than any pause the model was fitted on "
                     f"(max {rng['hours_since_end'][1] / 24:.1f} d).")
    if pd.notna(row.get("last_end")):
        for e in db.query("SELECT label, notes FROM episodes WHERE kind='non_fountaining' AND start_t > ?",
                          (int(row["last_end"]),)):
            notes.append(f"Non-fountaining eruptive event since the last episode ({e['label']}): {e['notes']}")
    return notes


def latest() -> dict:
    m = models()
    meta = (m or {}).get("meta") or {}
    fr = frame.load(int(time.time()) - 2 * 86400)
    if m is None or fr.empty:
        return {"model": None, "training": meta}
    fr[FEATURES] = fr[FEATURES].astype(float)
    newest = fr.iloc[-1]
    missing = [f for f in REQUIRED if pd.isna(newest[f])]
    pr = predictions(int(fr.index[0])).dropna()
    pr = pr[pr["t"].isin(fr.index)]
    t, ok = current_slot(fr, pr)
    row = fr.loc[[t]]
    feats = {f: (None if pd.isna(row.iloc[0][f]) else float(row.iloc[0][f])) for f in FEATURES}
    in_ep, why = _episode_in_progress(row.iloc[0])
    out = {
        "version": meta["version"], "t": t, "p12": None, "features": feats, "last_episode": row.iloc[0]["last_label"],
        "warnings": _warnings(row.iloc[0], meta), "in_episode": in_ep, "in_episode_reason": why,
        # the newest grid slot may be NULL if an input hasn't arrived yet
        "newest_slot": int(fr.index[-1]), "missing_now": missing,
    }
    if ok:
        last = pr[pr["t"] == t].iloc[0]
        contrib = m["fits"][-1].contributions(row)
        top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:5]
        trend = {}
        for hrs in (6, 24):
            prev = pr[pr["t"] <= t - hrs * 3600]
            if len(prev):
                trend[f"delta_{hrs}h"] = round(float(last["p12"] - prev.iloc[-1]["p12"]), 4)
        d6 = trend.get("delta_6h", 0.0)
        trend["arrow"] = "up" if d6 > 0.02 else ("down" if d6 < -0.02 else "flat")
        out.update({
            "p12": float(last["p12"]), "trend": trend,
            "top_factors": [{"feature": k, "label": LABELS.get(k, k), "value": feats.get(k), "effect": round(v, 4)}
                            for k, v in top if abs(v) > 1e-6],
        })
    else:  # no number: say why instead of showing an older one
        gone = [f for f in REQUIRED if feats[f] is None]
        out["reason"] = "episode" if row.iloc[0]["in_episode"] == 1 else ("feature_missing" if gone else "pending")
        out["missing"] = [{"feature": f, "label": LABELS.get(f, f)} for f in gone]
        if len(pr):
            out["last_valid"] = {"t": int(pr.iloc[-1]["t"]), "p12": float(pr.iloc[-1]["p12"])}
    return {"model": out,
            "training": {k: meta.get(k) for k in ("trained_at", "version", "horizon_h", "n_rows", "n_onsets", "cycles",
                                                  "train_start", "fit_through", "first_prediction", "refits",
                                                  "metrics", "required")}}
