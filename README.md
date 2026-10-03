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

On first start with an empty database the server runs `app/bootstrap.py` in the background:
it imports anything an older `data/volcano.db` already fetched, downloads the rest (≈1 h, mostly
UWE tremor history), then builds the grid and trains. Run it by hand with
`uv run python -m app.bootstrap`.

Optional `.env` keys: `FIRMS_MAP_KEY`, `DATA_DIR`, `DISABLE_SCHEDULER=1`.

## Architecture

```
sources (one table each, native resolution)
   └─ stitch.py ─▶ series_<type>: one authoritative 5-minute series per data type, src per slot
        └─ frame.py ─▶ frame: one row per 5-minute slot, all model features
             └─ model/service.py ─▶ pred_hazard, pred_ml: (t, p12, p24, p72) per slot
```

* **Grid:** 5-minute UTC slots from 2024-12-01 (three weeks before episode 1).
* **Stitching:** each slot takes the first source in priority order that has a value
  (`app/sources/__init__.py`). Sources that aren't directly comparable are mapped first, and the
  mapping is stored and shown on the dashboard.
* **None:** a model outputs NULL for a slot when any input it requires is missing (no
  imputation), or when an episode is under way.
* **Jobs:** `pipeline.update()` runs every 5 min: it re-stitches and recomputes frame and
  predictions from the earliest changed slot (at least the last 2 days). `pipeline.rebuild()`
  runs daily, or when the episode catalog changes: it rebuilds the whole grid, retrains, and
  re-infers.

### Sources

| Type | Source (priority order) | Table | Mapping onto the grid | Poll |
|---|---|---|---|---|
| tilt | UWD USGS 1-min release (az 300°) | `src_tilt_uwd_release` | 5-min means; authoritative | daily (≈60-day lag) |
| tilt | UWD HVO 2-day plot, digitized | `src_tilt_uwd_plot2d` | interpolated to slots; median offset to the series built so far | 10 min |
| tilt | UWD HVO 3-month plot, digitized | `src_tilt_uwd_plot3m` | same (aligned before the 2-day plot so that one can chain through it) | hourly |
| tilt | SDH USGS release | `src_tilt_sdh_release` | per-gap fit UWD ≈ a·E + b·N + c on ±5 days of pause data; used if R² ≥ 0.9 | daily |
| tremor | UWE HHZ (EarthScope FDSN) | `src_rsam_uwe` | 10-min RSAM, 1–5 Hz, µm/s; each window covers two slots | 10 min |
| tremor | UWE.QC HHZ | `src_rsam_uwe_qc` | as-is (agrees with UWE to ~1%) | 10 min |
| tremor | OBL, UWB, RIMD HHZ | `src_rsam_obl` … | fetched only around UWE gaps; scaled by the median UWE/substitute ratio within 24 h | on gaps |
| events | USGS ComCat | `earthquakes` | trailing 24 h counts (summit, all) | 5 min |
| events | HVO notices (HANS) | `notices` | precursory-activity keyword flag, valid 36 h | 12 min |
| events | HVO observatory messages ("Kilauea Message") | `messages` | display only (posts between Daily Updates) | 10 min |
| catalog | USGS episode table | `episodes` | episode start/end → repose, onset/trough tilt | 6 h |
| context | USGS status, NASA FIRMS, Open-Meteo | `raw_cache`, `firms` | display only | 10–30 min |

A plot digitization keeps a slot's first recorded value; only its last 6 hours may be revised.
That stops the hourly re-digitizing of a coarse plot from making history wobble.

### Notes

* **UWD tilt is not on EarthScope FDSN** (checked). Live tilt is digitized from HVO's PNG plots
  (`app/digitize.py`): the axis is read by OCR (per-label, with voting over "nice" tick steps)
  and the trace is extracted column by column.
* **Episode catalog:** comes from the USGS episode table. A snapshot ships in
  `app/episodes/seed_episodes.csv`, and manual overrides go in `manual_episodes.csv`. HVO notices
  can add the next episode provisionally before the table updates.
* **Genuine gaps** (both tiltmeters bad, or no seismometer reporting) stay empty. The dashboard
  shows each source's share of the grid, plus the gap share.

## Models (`app/model/`)

Both models are trained on the frame's on-the-hour rows during pauses since episode 4, using
rows where every feature is present. Features use only data at or before the slot: repose time, tilt recovery relative to the last deflation, tilt
relative to the last onset level, tilt rates (6 h least-squares; 24 h as the median of hourly
changes, so a few-hour step such as a dike intrusion doesn't inflate it for a day), RSAM level/trend, quake counts (exponentially weighted, 6 h e-folding,
so a swarm fades instead of dropping off after 24 h), and whether HVO's latest update reports
precursory activity (a flag that fades with the notice's age, 24 h e-folding).

* **hazard-1.0 (primary, left card)**. A discrete-time survival model: a logistic hourly hazard with splines on
  repose time and tilt recovery. Splines extrapolate as constants, so out-of-range states
  saturate. Tilt recovery is capped at the highest value seen at an actual onset (≈1.65×): above
  that, the only training hours come from pauses that hadn't ended, which taught the spline that
  more inflation means a lower hazard. It's projected forward 12/24/72 h with repose advancing and tilt inflating at the
  current rate. "Top factors" are per-feature log-odds contributions.
* **ml-1.0 (comparison, right card)**. Gradient-boosted trees, one per horizon, isotonic-calibrated on grouped
  out-of-fold predictions. Factors are ablations (the prediction change when a feature is set to
  its median).

Evaluation is grouped cross-validation by eruption cycle, reported for completed cycles and
including the current pause. It's shown in the UI and at `/api/probability` → `training.metrics`.
Models retrain daily, and the whole prediction grid is recomputed with the new model; the faint
line on the probability chart marks the training time (in-sample to its left).

Out-of-range conditions are flagged in the UI. Examples: tilt recovered well past every historical
onset level, a record-long pause, a non-fountaining vent event since the last episode.

## Dashboard controls

A sticky bar above the charts sets the time range (1, 7, 30 or 90 days, or **All**, meaning since the series began on Dec 23, 2024) for every timeseries:
onset probability, tilt, tilt rate, tremor and earthquakes. It also has a **Show eruption
episodes** toggle that shades each episode, plus the Sept 14 non-fountaining vent event, on all of
them. The browser remembers both choices. The tilt and tremor panels draw every source (thin)
under the stitched series (bold) and list each source's share of the grid and its mapping. The
"Model inputs" table shows the latest frame row and which inputs each model requires.

## API

* `/api/series/tilt?hours=` and `/api/series/tremor?hours=`: every source plus the stitched
  series, with mappings and grid shares
* `/api/frame?hours=`: feature frame
* `/api/predictions?hours=`: both prediction frames
* `/api/probability`: latest values, factors and warnings
* `/api/status`, `/api/notices`, `/api/earthquakes?days=`, `/api/episodes`, `/api/firms`,
  `/api/weather`, `/api/health` (per source, grouped)

Long ranges are thinned for display by keeping each bucket's min and max. OpenAPI docs are at
`/docs`.

## Tests

`uv run pytest`
