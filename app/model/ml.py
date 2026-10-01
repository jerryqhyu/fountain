"""ml-v1: gradient-boosted trees predicting onset-within-H directly (one model per horizon),
with isotonic calibration fitted on grouped out-of-fold predictions.

Trees extrapolate as constants, so out-of-range states fall back to the nearest seen leaf.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from ..frame import FEATURES

VERSION = "ml-1.0"


def _gbm() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=250, min_samples_leaf=60,
        l2_regularization=1.0, random_state=0,
    )


class MLModel:
    version = VERSION

    def __init__(self, horizons=(12, 24, 72)):
        self.horizons = tuple(horizons)

    def fit(self, df: pd.DataFrame, ys: dict[int, np.ndarray], groups: np.ndarray) -> "MLModel":
        self.models_, self.iso_ = {}, {}
        X = df[FEATURES].to_numpy(float)
        self.medians_ = df[FEATURES].median()
        for H in self.horizons:
            y = ys[H]
            ok = ~np.isnan(y)
            Xh, yh, gh = X[ok], y[ok].astype(int), groups[ok]
            oof = np.full(len(yh), np.nan)
            n_splits = min(5, len(np.unique(gh)))
            for tr, te in GroupKFold(n_splits=n_splits).split(Xh, yh, gh):
                m = _gbm().fit(Xh[tr], yh[tr])
                oof[te] = m.predict_proba(Xh[te])[:, 1]
            self.iso_[H] = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(oof, yh)
            self.models_[H] = _gbm().fit(Xh, yh)
        return self

    def predict(self, df: pd.DataFrame) -> dict[int, np.ndarray]:
        X = df[FEATURES].to_numpy(float)
        out = {H: self.iso_[H].predict(self.models_[H].predict_proba(X)[:, 1]) for H in self.horizons}
        # horizons are fitted separately; P(onset within H) must not decrease with H
        prev = None
        for H in sorted(self.horizons):
            if prev is not None:
                out[H] = np.maximum(out[H], prev)
            prev = out[H]
        return out

    def contributions(self, row: pd.DataFrame, H: int = 24) -> dict[str, float]:
        """Change in P(onset within H) when each feature is replaced by its training median."""
        base = float(self.predict(row)[H][0])
        out = {}
        for f in FEATURES:
            r2 = row.copy()
            r2[f] = self.medians_[f]
            out[f] = base - float(self.predict(r2)[H][0])
        return out
