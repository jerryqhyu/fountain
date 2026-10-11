from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from app import digitize
from app.config import HST
from app.episodes.catalog import parse_hst
from app.episodes.suggest import classify
from app.sources.tilt_release import az_component
from app.model.service import labels

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
    df = pd.DataFrame({"next_onset": [5 * 3600, np.nan, np.nan]}, index=[0, 3600, 7200])
    y = labels(df, 12, now=12 * 3600 + 3600)
    assert y[0] == 1.0          # onset 5 h later
    assert y[1] == 0.0          # 12 h elapsed with no onset
    assert np.isnan(y[2])       # horizon not yet elapsed -> censored


def test_labels_ignore_onsets_after_the_cutoff():
    # an onset 5 h ahead is in the catalog, but a model fitted at t = 2 h cannot know it yet
    df = pd.DataFrame({"next_onset": [5 * 3600.0] * 3}, index=[0, 3600, 7200])
    y = labels(df, 1, now=2 * 3600)
    assert y[0] == 0.0 and y[1] == 0.0  # their hour had passed without an onset
    assert np.isnan(y[2])


def test_month_starts_are_hst_month_boundaries():
    from app.model.service import month_starts
    assert month_starts(hst(2025, 6, 30, 23, 0)) == []  # warm-up: nothing is predicted before July 2025
    assert month_starts(hst(2025, 7, 1, 0, 0)) == [hst(2025, 7, 1)]
    got = month_starts(hst(2026, 1, 15))
    assert got[0] == hst(2025, 7, 1) and got[-1] == hst(2026, 1, 1) and len(got) == 7


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


def test_on_grid_uses_array_start():
    from app.stitch import on_grid
    g = np.arange(1790000000, 1790000000 + 36000, 3600)
    out = on_grid(g, np.array([1790007200, 1790018000]), np.array([1.0, 2.0]))
    assert out[2] == 1.0 and out[5] == 2.0 and np.isnan(out[0])


def test_hourly_slot_holds_only_the_hour_before_it():
    from app.stitch import hourly
    h = 1790002800  # a top of the hour
    # 5-minute means keyed by bin start: the 00:55 bin ends at the slot, the 01:00 bin belongs to the next
    d = pd.DataFrame({"t": [h - 3600, h - 300, h], "v": [1.0, 3.0, 9.0]})
    out = hourly(d, 300).set_index("t")["v"]
    assert out[h] == 2.0 and out[h + 3600] == 9.0
    # point samples: one taken exactly on the hour is already known at that slot
    out = hourly(pd.DataFrame({"t": [h - 1800, h, h + 300], "v": [1.0, 3.0, 9.0]}), 0).set_index("t")["v"]
    assert out[h] == 2.0 and out[h + 3600] == 9.0


def test_tremor_substitute_with_implausible_scale_is_rejected(monkeypatch):
    from app import stitch
    g = np.arange(1790002800, 1790002800 + 72 * 3600, 3600)
    gap = slice(30, 34)
    uwe = np.full(len(g), 0.25)
    uwe[gap] = np.nan
    # UWB is railed (thousands of µm/s whatever the ground does); RIMD is healthy at 0.8× UWE
    level = {"uwe": uwe, "uwb": np.full(len(g), 5000.0), "rimd": np.full(len(g), 0.2)}

    def source(key, cols="t, v"):
        v = level.get(key, np.full(len(g), np.nan))
        return pd.DataFrame({"t": g, "v": v}).dropna()

    monkeypatch.setattr(stitch, "source", source)
    v, src, info = stitch.build_tremor(g)
    assert set(src[gap]) == {"rimd"} and np.allclose(v[gap], 0.25)
    assert [(r["station"], r["used"]) for r in info["ratios"]] == [("uwb", False), ("rimd", True)]
    assert info["ratios"][0]["ratio"] == 5e-05
    # with no plausible substitute the slots stay a genuine gap, and the rejection is still recorded
    del level["rimd"]
    v, src, info = stitch.build_tremor(g)
    assert np.isnan(v[gap]).all() and not src[gap].any()
    assert [(r["station"], r["used"]) for r in info["ratios"]] == [("uwb", False)]


def test_substitute_fetch_does_not_count_a_dead_sensor_as_cover(monkeypatch):
    from app.sources import rsam
    level = {"src_rsam_uwe": 0.25, "src_rsam_uwb": 5000.0, "src_rsam_rimd": 0.2}

    def query(sql, params=()):
        v = next((v for table, v in level.items() if f"FROM {table} " in sql), None)
        return [] if v is None else [{"t": t, "v": v} for t in range(0, 6000, 600)]

    monkeypatch.setattr(rsam.db, "query", query)
    assert not rsam.plausible("uwb", 0, 6000)
    assert rsam.plausible("rimd", 0, 6000)
    assert rsam.plausible("obl", 0, 6000)  # no overlap to judge by: left to stitching


def test_robust_rate_ignores_short_step():
    import numpy as np
    import pandas as pd

    from app.frame import SLOTS_PER_H, robust_rate

    n = 48 * SLOTS_PER_H
    v = 0.1 * np.arange(n) / SLOTS_PER_H  # steady 0.1 µrad/h
    v[-12 * SLOTS_PER_H:] += 40.0  # 40 µrad step 12 h ago (e.g. a dike intrusion)
    r = robust_rate(pd.Series(v), 24).to_numpy()
    assert abs(r[-1] - 0.1) < 1e-9
