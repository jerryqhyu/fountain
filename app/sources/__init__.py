"""Source registry.

Every external input is a *source* with one fetch function and (for time series) one table.
Time-series sources are grouped by data type; within a type, the list order is the stitch
priority: each 5-minute grid slot takes its value from the first source that has one.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeriesSource:
    key: str          # also the source_health key
    table: str
    label: str
    note: str


TILT_SOURCES = [
    SeriesSource("uwd_release", "src_tilt_uwd_release", "UWD · USGS 1-min release",
                 "Authoritative borehole tilt, az 300°; published ~60 days late."),
    SeriesSource("uwd_plot2d", "src_tilt_uwd_plot2d", "UWD · HVO 2-day plot",
                 "Digitized every 10 min (~4 min/px, ~0.01 µrad/px); offset-aligned when stitched."),
    SeriesSource("uwd_plot3m", "src_tilt_uwd_plot3m", "UWD · HVO 3-month plot",
                 "Digitized hourly (~3 h/px, ~0.16 µrad/px); bridges the release lag."),
    SeriesSource("sdh_release", "src_tilt_sdh_release", "SDH · USGS release, fitted",
                 "Second summit tiltmeter; fills UWD gaps via a local linear fit (R² ≥ 0.9)."),
]

TREMOR_STATIONS = {  # key -> (station, location code)
    "uwe": ("UWE", ""),
    "uwe_qc": ("UWE", "QC"),
    "obl": ("OBL", ""),
    "uwb": ("UWB", ""),
    "rimd": ("RIMD", ""),
}
TREMOR_SOURCES = [
    SeriesSource("uwe", "src_rsam_uwe", "UWE HHZ", "Primary: caldera rim, 0.4 km from tiltmeter UWD."),
    SeriesSource("uwe_qc", "src_rsam_uwe_qc", "UWE.QC HHZ", "Co-located sensor; agrees with UWE to ~1%."),
    SeriesSource("obl", "src_rsam_obl", "OBL HHZ (scaled)",
                 "Fetched only to fill UWE gaps; scaled by the local UWE/OBL ratio."),
    SeriesSource("uwb", "src_rsam_uwb", "UWB HHZ (scaled)", "Second substitute for UWE gaps."),
    SeriesSource("rimd", "src_rsam_rimd", "RIMD HHZ (scaled)", "Third substitute for UWE gaps."),
]
SUBSTITUTE_TREMOR = ["obl", "uwb", "rimd"]

# Dashboard grouping of every source_health key.
HEALTH_GROUPS = [
    ("Official", [("usgs_status", "USGS Volcano API status"), ("hans", "HVO notices (HANS)"),
                  ("hvo_messages", "HVO observatory messages"),
                  ("usgs_episode_table", "USGS episode table")]),
    ("Tilt", [(s.key, s.label) for s in TILT_SOURCES]),
    ("Tremor", [(s.key, s.label) for s in TREMOR_SOURCES]),
    ("Earthquakes", [("comcat", "USGS ComCat")]),
    ("Context", [("firms", "NASA FIRMS"), ("weather", "Open-Meteo")]),
    ("Pipeline", [("stitch", "Stitch → grid"), ("frame", "Feature frame"), ("infer", "Model inference"),
                  ("train", "Model training")]),
]
