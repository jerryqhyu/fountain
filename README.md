# Kīlauea Fountaining Dashboard

A personal dashboard for the Halemaʻumaʻu episodic-fountaining eruption. Official USGS
status sits at the top; custom analytics and an **experimental, unofficial** estimate of the
chance a fountaining episode starts are at the bottom. Every data source is free.

## Run

```bash
uv sync
./scripts/run.sh                 # http://127.0.0.1:8000 (caffeinate keeps the Mac awake)
```

or directly: `uv run uvicorn app.main:app --host 127.0.0.1 --port 8000`.

Keep it running in the background and restart it if it crashes:

```bash
nohup ./scripts/serve_forever.sh > logs/server.log 2>&1 &
pkill -f serve_forever.sh; pkill -f "uvicorn app.main:app"   # stop
```

`scripts/install_launchagent.sh` installs a login LaunchAgent. It **only works if the project
isn't under a TCC-protected folder** like `~/Documents`, or if `/bin/zsh` has Full Disk Access,
because macOS blocks launchd jobs from reading `~/Documents`. Use `DASHBOARD_HOST` and
`DASHBOARD_PORT` to change the bind address. Don't use `HOST`, because zsh sets it to the machine name.

Docker (untested on this machine, which has no Docker): `docker compose up --build`.

On first start with an empty `data/`, the server backfills history on its own: notices, earthquakes,
tilt releases, the episode table, and about an hour of RSAM. It trains a first model without
RSAM, then retrains once RSAM is in. `scripts/backfill_rsam.py` runs the RSAM step by hand.

Optional `.env` keys: `FIRMS_MAP_KEY`, `DATA_DIR`, `DISABLE_SCHEDULER=1`, `STORE_DIR`, `STORE_AUTOEXPORT=0`.

## History and model store

`store/` is the durable record (Parquet + JSON, committed to git); `data/volcano.db` is a working copy rebuilt from it. On a fresh checkout the app loads `store/` into an empty DB at startup, so history isn't re-pulled. `uv run python -m app.store export|hydrate|verify|status` manage it. Layout, rules and sizes are in [docs/DATA_STORE.md](docs/DATA_STORE.md). The app never commits for you: `git add store && git commit && git push`.

## Data sources and how they're used

| Source | Endpoint | Poll |
|---|---|---|
| USGS Volcano API `vhpstatus` | alert level, color code | 10 min |
| USGS HANS (`search/preflight`, `search/search` POST, the same calls the HANS search page makes) | all HVO Kīlauea notices, full text | 12 min |
| USGS ComCat FDSN event | earthquakes, 19.1–19.6 N, 155.6–154.8 W | 5 min |
| EarthScope FDSN dataselect (ObsPy) | `HV.UWE..HHZ` → 10-min RSAM, 1–5 Hz | 10 min |
| HVO UWD tilt **plot PNGs** (digitized) | live tilt | 10 min / hourly |
| USGS ScienceBase tilt releases (2024, 2025 H1, Jul 2025→present at 60-day lag) | 1-min UWD tilt history | daily |
| USGS "Eruption Information" episode table (scraped) | episode catalog | 6 h |
| NASA FIRMS area API | VIIRS/MODIS hotspots over the caldera | 30 min |
| Open-Meteo | summit weather/visibility | 30 min |

Every call is wrapped: failures are logged to `source_health`, the last good data keeps being
served, and the UI shows per-source freshness and stale/error chips.

### Things that differ from the original spec, and why

* **UWD tilt is not on EarthScope FDSN.** The station service has no HV.UWD channels (checked).
  USGS publishes tilt data only as CSV releases with about 60 days of lag, plus PNG plots that refresh
  every 10 minutes. So the live tail comes from **digitizing the plots**
  (`app/analytics/digitize.py`): tesseract OCR reads the axis labels, and the trace is picked out
  column by column. Each plot is then offset-aligned onto the 1-minute CSV baseline. Alignment
  error is about 0.03–0.06 µrad (see `tilt_stitch` in `/api/tilt`). A Wayback Machine copy of the
  3-month plot bridges the June 2026 gap between the release and the live plots.
* **Episode catalog seed.** The catalog comes from the official USGS episode table (episodes 1–54
  plus the non-fountaining vent event of 14 Sep 2026), not from hand-seeding. A snapshot ships in
  `app/episodes/seed_episodes.csv`. Put manual overrides in `app/episodes/manual_episodes.csv`.
  The HANS keyword parser posts candidate new entries to `/api/episodes` → `suggestions`.
* **Tremor station.** RSAM uses UWE (0.4 km from UWD) rather than a named HVO RSAM product.

## Models (`app/model/`)

Both models are trained on every hourly sample during pauses since episode 4. The features use
only data up to that hour: repose time, tilt recovery relative to the last deflation, tilt
relative to the last onset level, tilt rates, RSAM level/trend, quake counts, and whether HVO's
latest update reports precursory activity.

* **hazard-1.0 (primary, left card)**. A discrete-time survival model: a logistic hourly hazard with splines on
  repose time and tilt recovery. Splines extrapolate as constants, so out-of-range states
  saturate. It's projected forward 12/24/72 h with repose advancing and tilt inflating at the
  current rate. "Top factors" are per-feature log-odds contributions.
* **ml-1.0 (comparison, right card)**. Gradient-boosted trees, one per horizon, isotonic-calibrated on grouped
  out-of-fold predictions. Factors are ablations (the prediction change when a feature is set to
  its median).

Evaluation is grouped cross-validation by eruption cycle, reported for completed cycles and
including the current pause. It's shown in the UI and at `/api/probability` → `training.metrics`.
Models retrain daily. Predictions run every 10 minutes and are logged to `probability_log`.

Out-of-range conditions are flagged in the UI. Examples: tilt recovered well past every historical
onset level, a record-long pause, a non-fountaining vent event since the last episode.

## Dashboard controls

A sticky bar above the charts sets the time range (1, 7, 30 or 90 days, or **All**, meaning since the series began on Dec 23, 2024) for every timeseries:
onset probability, tilt, tilt rate, tremor and earthquakes. It also has a **Show eruption
episodes** toggle that shades each episode, plus the Sept 14 non-fountaining vent event, on all of
them. The browser remembers both choices. In the probability chart, faded lines are hindcasts
(the current models re-run on past hours, so they're in-sample) and solid lines were logged live.

## API

`/api/status`, `/api/notices?limit=5`, `/api/tilt?hours=72`, `/api/tremor?hours=72`,
`/api/earthquakes?days=7`, `/api/episodes`, `/api/probability`, `/api/probability/history`,
`/api/firms`, `/api/weather`, `/api/health`. OpenAPI docs are at `/docs`.

## Tests

`uv run pytest`
