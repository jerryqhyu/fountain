"""Runtime configuration. Everything optional comes from .env."""
from __future__ import annotations

import os
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "volcano.db"))
STATIC_DIR = ROOT / "static"

FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY", "").strip()
# Set DISABLE_SCHEDULER=1 to run the API without background polling (tests).
DISABLE_SCHEDULER = os.getenv("DISABLE_SCHEDULER", "0") == "1"

HST = ZoneInfo("Pacific/Honolulu")

# Kīlauea identifiers (verified against the Volcano API: vnum 332010, volcanoCd hi3).
VNUM = "332010"
VOLCANO_CD = "hi3"
SUMMIT_LAT, SUMMIT_LON = 19.421, -155.287

# Earthquake search box: summit caldera + upper East and Southwest rift zones.
QUAKE_BOX = dict(minlatitude=19.10, maxlatitude=19.60, minlongitude=-155.60, maxlongitude=-154.80)
# Sub-regions used for rate breakdowns (checked in order; first match wins).
QUAKE_REGIONS = [
    ("summit", 19.36, 19.46, -155.33, -155.22),
    ("upper_erz", 19.30, 19.42, -155.22, -155.00),
    ("swrz", 19.20, 19.40, -155.45, -155.30),
]

# Seismic station for tremor/RSAM. UWE is 0.4 km from tiltmeter UWD on the caldera rim.
TREMOR_NET, TREMOR_STA, TREMOR_LOC, TREMOR_CHA = "HV", "UWE", "", "HHZ"
FDSN_BASE = "https://service.earthscope.org"
RSAM_WINDOW_S = 600
RSAM_BAND = (1.0, 5.0)

# Tilt: UWD az-300 is the radial component HVO uses for episode forecasting.
TILT_STATION = "UWD"
TILT_AZIMUTH_DEG = 300.0

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130 Safari/537.36 kilauea-dashboard/0.1 (personal, non-commercial)"
)
