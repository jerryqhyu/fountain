from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from app.analytics import digitize
from app.config import HST
from app.episodes.catalog import parse_hst
from app.episodes.suggest import classify
from app.fetchers.tilt_release import az_component
from app.model.features import labels

FIX = Path(__file__).parent / "fixtures"


def hst(*a):
    return int(datetime(*a, tzinfo=HST).timestamp())


def test_parse_hst_variants():
    assert parse_hst("January 24, 2025 - 11:28 p.m.") == hst(2025, 1, 24, 23, 28)
    assert parse_hst("December 23, 2024 - 4 p.m.") == hst(2024, 12, 23, 16, 0)
    assert parse_hst("August 25, 2026 - 10:30 a.m.") == hst(2026, 8, 25, 10, 30)
    assert parse_hst("July 9 - 1:20 p.m.", default_year=2025) == hst(2025, 7, 9, 13, 20)
    assert parse_hst("December 26, 2024 - 12:15 a.m.") == hst(2024, 12, 26, 0, 15)
    assert parse_hst("TBD") is None


def test_az300_projection():
    # az 300 = N60W: pure west tilt projects positively, pure east negatively
    assert az_component(np.array([-1.0]), np.array([0.0]))[0] > 0.86
    assert abs(az_component(np.array([0.0]), np.array([1.0]))[0] - 0.5) < 1e-9


def test_digitize_month_plot():
    d = digitize.digitize((FIX / "UWD-TILT-month.png").read_bytes(), expect_span_s=30 * 86400,
                          fallback_range=(0, 30 * 86400))
    assert 700 < len(d.t) < 760
    assert d.y_fit_resid < 0.1
    assert abs((d.end - d.start) - 30 * 86400) < 3600
    # the legend's blue sample line must not leak into the trace
    assert np.max(d.vmax - d.v) < 2.0
    assert -12 < d.v.min() < -9 and 3 < d.v.max() < 6


def test_labels_censoring():
    df = pd.DataFrame({"t": [0, 3600, 7200], "next_onset": [5 * 3600, np.nan, np.nan]})
    y = labels(df, 12, now=12 * 3600 + 3600)
    assert y[0] == 1.0          # onset 5 h later
    assert y[1] == 0.0          # 12 h elapsed with no onset
    assert np.isnan(y[2])       # horizon not yet elapsed -> censored


def test_classify_precursor_not_onset():
    txt = "Episode 55 precursory overflows started from the north vent at 12:45 p.m."
    phases = {p for _, p, _ in classify(txt)}
    assert "onset" not in phases and "precursor" in phases


def test_parse_event_time():
    from app.episodes.suggest import parse_event_time
    sent = hst(2026, 8, 26, 12, 0)
    assert parse_event_time("Episode 54 ... ended abruptly at 7:33 p.m. HST on August 25.", sent) == ("end", hst(2026, 8, 25, 19, 33))
    assert parse_event_time("Episode 54 began at 10:30 HST on August 25, 2026.", sent) == ("onset", hst(2026, 8, 25, 10, 30))
    assert parse_event_time("Episode 54 began at 10:30 a.m. HST on Tuesday, August 25", sent) == ("onset", hst(2026, 8, 25, 10, 30))
    assert parse_event_time("began at 10:30 a.m. HST on August 1", sent) is None  # too old for this notice
