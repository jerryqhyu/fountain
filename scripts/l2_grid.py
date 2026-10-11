"""Walk-forward grid search over the hazard model's L2 penalty.

    uv run python -m scripts.l2_grid [l2 ...]

For each penalty, every month is predicted by a model fitted only on data before it (as in
production). Reported per penalty: log loss and Brier skill on completed cycles, how much
consecutive fits disagree on the same hours (median absolute change in the log-odds of the
12 h probability), and the mean probability over the still-open pause. The last line gives the
range to choose from: the penalty with the best log loss, and the strongest one still within one
standard error of it (bootstrap over cycles). hazard.L2 is picked inside that range.
"""
from __future__ import annotations

import functools
import sys
import time

import numpy as np
import pandas as pd

from app import frame
from app.frame import FEATURES
from app.model import service
from app.model.hazard import HazardModel

GRID = [1, 2, 3.3, 5, 6.7, 10, 14, 20, 33, 50, 100]


def evaluate(l2: float, df: pd.DataFrame, now: int) -> dict:
    service.HazardModel = functools.partial(HazardModel, l2)
    fits = [hz for c in service.month_starts(now) if (hz := service.fit_month(df, c)) is not None]
    out = df[df.index >= fits[0].fit_info_["cutoff"]]
    p = service.predict(fits, out)
    y = service.labels(out, service.HORIZON_H, now)
    open_pause = np.isnan(out["next_onset"].to_numpy(float))
    ok = ~open_pause & ~np.isnan(y) & ~np.isnan(p)
    q = np.clip(p, 1e-4, 1 - 1e-4)
    loss = -(y * np.log(q) + (1 - y) * np.log(1 - q))
    by_cycle = pd.DataFrame({"c": out["last_label"].to_numpy()[ok], "loss": loss[ok]}).groupby("c")["loss"].agg(["sum", "size"])
    same = df[service.valid_mask(df) & df["next_onset"].notna().to_numpy()]  # hours every fit can be compared on
    logit = [np.log(v / (1 - v)) for v in (np.clip(hz.predict(same)[service.HORIZON_H], 1e-6, 1 - 1e-6) for hz in fits)]
    return {"l2": l2, "log_loss": loss[ok].mean(),
            "brier_skill": 1 - np.mean((q[ok] - y[ok]) ** 2) / np.mean((y[ok].mean() - y[ok]) ** 2),
            "refit_change": float(np.median([np.median(np.abs(b - a)) for a, b in zip(logit, logit[1:])])),
            "open_pause_mean": float(np.nanmean(p[open_pause])) if open_pause.any() else np.nan, "by_cycle": by_cycle}


def main() -> None:
    grid = [float(a) for a in sys.argv[1:]] or GRID
    now = int(time.time())
    df = frame.load()
    df[FEATURES] = df[FEATURES].astype(float)
    rows = [evaluate(l2, df, now) for l2 in grid]
    best = min(rows, key=lambda r: r["log_loss"])
    c = best["by_cycle"]
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(c), (2000, len(c)))
    se = float(np.std(c["sum"].to_numpy()[draws].sum(axis=1) / c["size"].to_numpy()[draws].sum(axis=1)))
    upper = max((r for r in rows if r["log_loss"] <= best["log_loss"] + se), key=lambda r: r["l2"])
    print(pd.DataFrame([{k: v for k, v in r.items() if k != "by_cycle"} for r in rows]).round(4).to_string(index=False))
    print(f"best log loss at L2={best['l2']:g} ({best['log_loss']:.4f} ± {se:.4f} over {len(c)} cycles); "
          f"strongest penalty within one standard error: L2={upper['l2']:g}; in use: L2={HazardModel().l2:g}")


if __name__ == "__main__":
    main()
