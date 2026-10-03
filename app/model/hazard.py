"""hazard-v1: discrete-time survival model (hourly logistic hazard).

h(x_t) = P(onset in (t, t+1h] | no onset yet, x_t), fitted by penalized logistic regression
on every pause-hour since episode 4. Nonlinear effects of repose time and tilt recovery
enter through natural-ish cubic splines with *constant extrapolation*, so states outside
the training range (e.g. a record-long pause) saturate instead of blowing up. Tilt recovery is
capped at the highest value seen at an onset, so the effect saturates there instead of turning down.

P(onset within H) = 1 - prod_{k<H} (1 - h(x_{t+k})), where the future path holds the
current state fixed except that repose time advances and tilt keeps inflating at the
current 24 h rate (clipped).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler

VERSION = "hazard-1.0"

GROUPS = {  # design-matrix blocks -> source feature
    "hours_since_end": "hours_since_end",
    "recovery_ratio": "recovery_ratio",
    "tilt_rate_24h": "tilt_rate_24h",
    "rsam_ratio_log": "rsam_ratio_log",
    "precursor": "precursor",
    "eq_summit_ew": "eq_summit_ew",
}
CLIP = {
    "recovery_ratio": (-0.5, 2.0),
    "tilt_rate_24h": (-0.3, 0.3),
    "rsam_ratio_log": (-1.0, 1.0),
}


class HazardModel:
    version = VERSION

    def __init__(self, C: float = 0.3):
        self.C = C

    # --- design matrix -------------------------------------------------------
    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        x = pd.DataFrame(index=df.index)
        x["hours_since_end"] = np.log(np.clip(df["hours_since_end"].to_numpy(float), 1.0, None))
        for k in ("recovery_ratio", "tilt_rate_24h", "rsam_ratio_log"):
            lo, hi = CLIP[k]
            if k == "recovery_ratio":
                hi = min(hi, self.rr_cap_)
            x[k] = np.clip(df[k].to_numpy(float), lo, hi)
        x["precursor"] = df["precursor"].to_numpy(float)
        x["eq_summit_ew"] = np.log1p(df["eq_summit_ew"].to_numpy(float))
        return x

    def _design(self, x: pd.DataFrame, fit: bool = False) -> tuple[np.ndarray, list[str]]:
        if fit:
            self.medians_ = x.median()
        x = x.fillna(self.medians_)
        blocks, names = [], []
        for col in ("hours_since_end", "recovery_ratio"):
            if fit:
                sp = SplineTransformer(n_knots=5, degree=3, knots="quantile", extrapolation="constant",
                                       include_bias=False)
                sp.fit(x[[col]])
                setattr(self, f"spline_{col}_", sp)
            b = getattr(self, f"spline_{col}_").transform(x[[col]])
            blocks.append(b)
            names += [col] * b.shape[1]
        for col in ("tilt_rate_24h", "rsam_ratio_log", "precursor", "eq_summit_ew"):
            blocks.append(x[[col]].to_numpy(float))
            names.append(col)
        X = np.hstack(blocks)
        if fit:
            self.scaler_ = StandardScaler().fit(X)
        return self.scaler_.transform(X), names

    # --- fit / predict ---------------------------------------------------------
    def fit(self, df: pd.DataFrame, y1: np.ndarray) -> "HazardModel":
        # Cap recovery at the highest level seen at an actual onset. Beyond it the only training
        # hours come from pauses that hadn't ended yet (e.g. the current record pause), which would
        # teach the spline that *more* inflation means a *lower* hazard; capped, it saturates.
        pos = df["recovery_ratio"].to_numpy(float)[y1.astype(bool)]
        self.rr_cap_ = float(np.nanmax(pos)) if np.isfinite(pos).any() else CLIP["recovery_ratio"][1]
        x = self._prep(df)
        X, self.names_ = self._design(x, fit=True)
        self.clf_ = LogisticRegression(C=self.C, max_iter=5000)
        self.clf_.fit(X, y1.astype(int))
        self.train_range_ = {c: (float(df[c].min()), float(df[c].max())) for c in GROUPS}
        self.x_mean_design_ = X.mean(axis=0)
        return self

    def hazard(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = self._design(self._prep(df))
        return self.clf_.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame, horizons=(12, 24, 72)) -> dict[int, np.ndarray]:
        Hmax = max(horizons)
        n = len(df)
        base = df[list(GROUPS) + ["last_deflation_urad"]].reset_index(drop=True)
        rate = np.clip(base["tilt_rate_24h"].fillna(0).to_numpy(float), 0.0, 0.3)
        defl = base["last_deflation_urad"].to_numpy(float)
        ks = np.arange(Hmax)
        fut = base.loc[np.repeat(np.arange(n), Hmax)].reset_index(drop=True)
        kk = np.tile(ks, n)
        fut["hours_since_end"] = fut["hours_since_end"].to_numpy(float) + kk
        with np.errstate(invalid="ignore", divide="ignore"):
            dr = np.where(defl > 0.5, rate / defl, 0.0)
        fut["recovery_ratio"] = fut["recovery_ratio"].to_numpy(float) + np.repeat(dr, Hmax) * kk
        h = self.hazard(fut).reshape(n, Hmax)
        logsurv = np.cumsum(np.log1p(-np.clip(h, 0, 1 - 1e-9)), axis=1)
        return {H: 1 - np.exp(logsurv[:, H - 1]) for H in horizons}

    def contributions(self, row: pd.DataFrame) -> dict[str, float]:
        """Log-odds contribution of each feature to the current hourly hazard vs. the average pause-hour."""
        X, names = self._design(self._prep(row))
        coef = self.clf_.coef_[0]
        d = (X[0] - self.x_mean_design_) * coef
        out: dict[str, float] = {}
        for nm, v in zip(names, d):
            out[nm] = out.get(nm, 0.0) + float(v)
        return out
