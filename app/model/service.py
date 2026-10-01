"""Train both models on the feature frame and run them over the grid.

Predictions are frames with the same layout for both models: pred_<model>(t, p12, p24, p72),
one row per 5-minute slot. A slot is NULL when an episode is under way or when any input the
model requires is genuinely missing (no imputation).
"""
from __future__ import annotations

import logging
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from .. import db, frame
from ..config import DATA_DIR
from ..frame import FEATURES, LABELS
from ..sources.base import tracked
from .hazard import GROUPS, HazardModel
from .ml import MLModel

log = logging.getLogger("model")
HORIZONS = (12, 24, 72)
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
FIRST_TRAIN_EPISODE = 4  # episodes 1–3 were long/continuous, a different style
REQUIRED = {"hazard": list(GROUPS), "ml": list(FEATURES)}
PRED_TABLE = {"hazard": "pred_hazard", "ml": "pred_ml"}


# ------------------------------------------------------------------ training
def labels(df: pd.DataFrame, horizon_h: float, now: int) -> np.ndarray:
    """1 if the next onset is within (t, t+H]; 0 if not; NaN if not yet knowable."""
    h = horizon_h * 3600
    t = df.index.to_numpy(float)
    nxt = df["next_onset"].to_numpy(float)
    y = np.full(len(df), np.nan)
    known = ~np.isnan(nxt)
    y[known] = ((nxt[known] - t[known]) <= h).astype(float)
    y[~known & (t + h <= now)] = 0.0
    return y


def training_frame(now: int) -> tuple[pd.DataFrame, dict]:
    df = frame.load()
    ep4 = db.query_one("SELECT end_t FROM episodes WHERE num=?", (FIRST_TRAIN_EPISODE,))["end_t"]
    df = df[(df.index % 3600 == 0) & (df.index > ep4) & (df.index < now - 3600)]
    df = df[(df["in_episode"] == 0) & df[FEATURES].notna().all(axis=1)].copy()
    df["cycle"] = df["last_label"].astype(str)
    return df, {H: labels(df, H, now) for H in (1,) + HORIZONS}


def _metrics(y: np.ndarray, p: np.ndarray, base_rate: float) -> dict:
    ok = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[ok].astype(int), np.clip(p[ok], 1e-4, 1 - 1e-4)
    if not len(y):
        return {}
    b, bc = brier_score_loss(y, p), brier_score_loss(y, np.full(len(y), base_rate))
    out = {"n": int(len(y)), "positives": int(y.sum()), "brier": round(float(b), 4),
           "brier_climatology": round(float(bc), 4),
           "brier_skill": round(float(1 - b / bc), 3) if bc > 0 else None,
           "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 4)}
    if 0 < y.sum() < len(y):
        out["auc"] = round(float(roc_auc_score(y, p)), 3)
    return out


def evaluate(df: pd.DataFrame, ys: dict, n_splits: int = 8) -> dict:
    """Out-of-fold skill, grouped by eruption cycle; reported for completed cycles and all."""
    groups = df["cycle"].to_numpy()
    oof = {m: {H: np.full(len(df), np.nan) for H in HORIZONS} for m in PRED_TABLE}
    for tr, te in GroupKFold(n_splits=n_splits).split(df, groups=groups):
        dtr, dte = df.iloc[tr], df.iloc[te]
        ok = ~np.isnan(ys[1][tr])
        hz = HazardModel().fit(dtr[ok], ys[1][tr][ok])
        for H, p in hz.predict(dte, HORIZONS).items():
            oof["hazard"][H][te] = p
        mlm = MLModel(HORIZONS).fit(dtr, {H: ys[H][tr] for H in HORIZONS}, groups[tr])
        for H, p in mlm.predict(dte).items():
            oof["ml"][H][te] = p
    open_cycle = df["cycle"].iloc[-1]
    closed = (df["cycle"] != open_cycle).to_numpy() if np.isnan(df["next_onset"].iloc[-1]) else np.ones(len(df), bool)
    res = {}
    for m in oof:
        res[m] = {"all": {}, "closed": {}}
        for H in HORIZONS:
            y = ys[H]
            res[m]["all"][str(H)] = _metrics(y, oof[m][H], float(np.nanmean(y)))
            yc = np.where(closed, y, np.nan)
            res[m]["closed"][str(H)] = _metrics(yc, oof[m][H], float(np.nanmean(yc)))
    return res


@tracked("train")
def train() -> str:
    t_start = time.time()
    now = int(t_start)
    df, ys = training_frame(now)
    ok = ~np.isnan(ys[1])
    hz = HazardModel().fit(df[ok], ys[1][ok])
    mlm = MLModel(HORIZONS).fit(df, {H: ys[H] for H in HORIZONS}, df["cycle"].to_numpy())
    onset_rows = df[ys[1] == 1]
    meta = {
        "trained_at": now,
        "n_rows": int(len(df)), "n_onsets": int(np.nansum(ys[1])), "cycles": int(df["cycle"].nunique()),
        "train_start": int(df.index.min()),
        "metrics": evaluate(df, ys),
        "feature_ranges_at_onset": {f: [float(onset_rows[f].min()), float(onset_rows[f].max())]
                                    for f in FEATURES if onset_rows[f].notna().any()},
        "versions": {"hazard": hz.version, "ml": mlm.version},
        "required": REQUIRED,
        "train_seconds": round(time.time() - t_start, 1),
    }
    joblib.dump({"hazard": hz, "ml": mlm, "meta": meta}, MODEL_DIR / "current.joblib")
    db.kv_set("model_meta", meta)
    return f"{meta['n_rows']} hourly rows, {meta['n_onsets']} onsets, {meta['train_seconds']} s"


_cache: dict | None = None


def models() -> dict | None:
    """Current model bundle, reloaded if another process retrained it."""
    global _cache
    p = MODEL_DIR / "current.joblib"
    if not p.exists():
        return None
    mtime = p.stat().st_mtime
    if _cache is None or _cache.get("_mtime") != mtime:
        _cache = joblib.load(p)
        _cache["_mtime"] = mtime
    return _cache


# ------------------------------------------------------------------ inference
def valid_mask(df: pd.DataFrame, model: str) -> np.ndarray:
    return ((df["in_episode"] == 0) & df[REQUIRED[model]].notna().all(axis=1)).to_numpy()


@tracked("infer")
def infer(t0: int | None = None) -> str:
    m = models()
    if m is None:
        raise RuntimeError("no trained model yet")
    df = frame.load(t0)
    if df.empty:
        return "no frame rows"
    df[FEATURES] = df[FEATURES].astype(float)
    out = []
    for name in PRED_TABLE:
        ok = valid_mask(df, name)
        p = {H: np.full(len(df), np.nan) for H in HORIZONS}
        idx = np.where(ok)[0]
        for c in range(0, len(idx), 5000):  # hazard projects 72 h per row: chunk to bound memory
            sel = idx[c:c + 5000]
            res = m[name].predict(df.iloc[sel], HORIZONS) if name == "hazard" else m[name].predict(df.iloc[sel])
            for H in HORIZONS:
                p[H][sel] = res[H]
        rows = [(int(t), *[None if np.isnan(p[H][i]) else float(p[H][i]) for H in HORIZONS])
                for i, t in enumerate(df.index)]
        db.upsert(PRED_TABLE[name], ["t", "p12", "p24", "p72"], rows)
        out.append(f"{name}: {int(ok.sum())}/{len(df)} slots valid")
    return "; ".join(out)


def predictions(model: str, t0: int, t1: int | None = None) -> pd.DataFrame:
    rows = db.query(f"SELECT t, p12, p24, p72 FROM {PRED_TABLE[model]} WHERE t>=? AND t<=? ORDER BY t",
                    (t0, t1 or 2**62))
    return pd.DataFrame(rows, columns=["t", "p12", "p24", "p72"])


# ------------------------------------------------------------------ latest, for the cards
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
                "The usual tilt threshold hasn't produced an onset this time, so both models are extrapolating.")
    rng = meta.get("feature_ranges_at_onset", {})
    hs = row.get("hours_since_end")
    if pd.notna(hs) and "hours_since_end" in rng and hs > rng["hours_since_end"][1]:
        notes.append(f"This pause ({hs / 24:.1f} d) is longer than any previous pause "
                     f"(max {rng['hours_since_end'][1] / 24:.1f} d).")
    if pd.notna(row.get("last_end")):
        for e in db.query("SELECT label, notes FROM episodes WHERE kind='non_fountaining' AND start_t > ?",
                          (int(row["last_end"]),)):
            notes.append(f"Non-fountaining eruptive event since the last episode ({e['label']}): {e['notes']}")
    return notes


def latest() -> dict:
    m = models()
    meta = (m or {}).get("meta") or db.kv_get("model_meta", {}) or {}
    fr = frame.load(int(time.time()) - 2 * 86400)
    if m is None or fr.empty:
        return {"models": {}, "training": meta}
    fr[FEATURES] = fr[FEATURES].astype(float)
    newest = fr.iloc[-1]
    out = {}
    for name in PRED_TABLE:
        pr = predictions(name, int(fr.index[0])).dropna()
        missing = [f for f in REQUIRED[name] if pd.isna(newest[f])]
        if pr.empty:
            out[name] = {"version": meta["versions"][name], "p": None, "missing_now": missing}
            continue
        last = pr.iloc[-1]
        t = int(last["t"])
        row = fr.loc[[t]]
        if name == "hazard":
            contrib, unit = m["hazard"].contributions(row), "log-odds of hourly hazard"
        else:
            contrib, unit = m["ml"].contributions(row, 24), "Δ P(24 h)"
        feats = {f: (None if pd.isna(row.iloc[0][f]) else float(row.iloc[0][f])) for f in FEATURES}
        top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:5]
        trend = {}
        for hrs in (6, 24):
            prev = pr[pr["t"] <= t - hrs * 3600]
            if len(prev):
                trend[f"delta_{hrs}h"] = round(float(last["p24"] - prev.iloc[-1]["p24"]), 4)
        d6 = trend.get("delta_6h", 0.0)
        trend["arrow"] = "up" if d6 > 0.02 else ("down" if d6 < -0.02 else "flat")
        in_ep, why = _episode_in_progress(row.iloc[0])
        out[name] = {
            "version": meta["versions"][name], "t": t, "p": {str(H): float(last[f"p{H}"]) for H in HORIZONS},
            "trend": trend, "features": feats, "last_episode": row.iloc[0]["last_label"],
            "top_factors": [{"feature": k, "label": LABELS.get(k, k), "value": feats.get(k), "effect": round(v, 4),
                             "unit": unit} for k, v in top if abs(v) > 1e-6],
            "warnings": _warnings(row.iloc[0], meta), "in_episode": in_ep, "in_episode_reason": why,
            # the newest grid slot may be NULL if an input hasn't arrived yet
            "newest_slot": int(fr.index[-1]), "missing_now": missing,
        }
    return {"primary": "hazard", "models": out,
            "training": {k: meta.get(k) for k in ("trained_at", "n_rows", "n_onsets", "cycles", "train_start",
                                                  "metrics", "versions", "required")}}
