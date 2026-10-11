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
UWE tremor history), then builds the grid and trains. If the server restarts before it has
finished, the next start resumes it. Run it by hand with `uv run python -m app.bootstrap`.

Optional `.env` keys: `FIRMS_MAP_KEY`, `DATA_DIR`, `DISABLE_SCHEDULER=1`.

## Architecture

```
sources (one table each, native resolution)
   └─ stitch.py ─▶ series_<type>: one authoritative hourly series per data type, src per slot
        └─ frame.py ─▶ frame: one row per hourly slot, all model features
             └─ model/service.py ─▶ pred_hazard: (t, p12) per slot from July 2025
```

* **Grid:** hourly UTC slots from 2024-12-01 (three weeks before episode 1). Tilt, tremor, the
  feature frame and the predictions all share it. A slot holds the hour that *ends* at its
  timestamp, so a row never contains data from after its own time.
* **Stitching:** every source is first reduced to the grid (the median of its samples in the
  slot's hour). Each slot then takes the first source in priority order that has a value
  (`app/sources/__init__.py`). Sources that aren't directly comparable are mapped first, and the
  mapping is stored and shown on the dashboard.
* **None:** the model outputs NULL for a slot when any input it requires is missing (no
  imputation), or when an episode is under way. The dashboard shades those hours in the
  probability chart ("feature missing") and, once the newest two slots both lack a value,
  shows a shaded tile naming the missing input instead of the last older probability.
* **Jobs:** `pipeline.update()` runs every 5 min: it re-stitches and recomputes frame and
  predictions from the earliest changed slot (at least the last 2 days). `pipeline.rebuild()`
  runs daily and rebuilds the whole grid. The models are refitted only when a new month begins
  or the episode catalog changes (see Model).

### Sources

| Type | Source (priority order) | Table | Mapping onto the grid | Poll |
|---|---|---|---|---|
| tilt | UWD USGS 1-min release (az 300°) | `src_tilt_uwd_release` | stored as 5-min means exactly as published; authoritative. Where USGS restarts a file after re-levelling the tiltmeter, the baseline step is removed when stitching | daily (≈60-day lag) |
| tilt | UWD HVO 2-day plot, digitized | `src_tilt_uwd_plot2d` | stored as 5-min samples; median offset to the series built so far | 10 min |
| tilt | UWD HVO 3-month plot, digitized | `src_tilt_uwd_plot3m` | same (aligned before the 2-day plot so that one can chain through it) | hourly |
| tilt | SDH USGS release | `src_tilt_sdh_release` | per-gap fit UWD ≈ a·E + b·N + c on ±5 days of pause data; used if R² ≥ 0.9 | daily |
| tremor | UWE HHZ (EarthScope FDSN) | `src_rsam_uwe` | 10-min RSAM, 1–5 Hz, µm/s; a slot is the median of its hour's windows | 10 min |
| tremor | UWE.QC HHZ | `src_rsam_uwe_qc` | as-is (agrees with UWE to ~1%) | 10 min |
| tremor | OBL, UWB, RIMD HHZ | `src_rsam_obl` … | fetched only around UWE gaps; scaled by the median UWE/substitute ratio within 24 h, rejected if that ratio is outside 0.1–10 (dead sensor) | on gaps |
| events | USGS ComCat | `earthquakes` | exponentially weighted summit count (6 h e-folding); the last 30 days are re-pulled each time so reviews and deletions carry over | 5 min |
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

## Model (`app/model/`)

One model, **hazard-1.1**, gives the probability that a fountaining episode starts within the
next 12 hours. It is trained on the frame's hourly rows during pauses since episode 4 (episodes
1–3 were near-continuous), using rows where every input is present. Inputs use only data at or
before the slot: repose time, tilt recovery relative to the last deflation, the 24 h tilt rate
(the median of hourly changes, so a few-hour step such as a dike intrusion doesn't inflate it
for a day), tremor in the last hour relative to the last 24 h, the summit quake count
(exponentially weighted, 6 h e-folding, so a swarm fades instead of dropping off after 24 h), and
whether HVO's latest update reports precursory activity (a flag that fades with the notice's
age, 24 h e-folding).

The fit is an L2-penalized logistic regression. The penalty weight (`L2` in `app/model/hazard.py`,
currently 20) comes from a walk-forward grid search, `uv run python -m scripts.l2_grid`: the
strongest penalty whose log loss on completed cycles is within one standard error of the best.
Changing it, or the model version, makes the next pipeline run refit every month.

It is a discrete-time survival model: a logistic hourly hazard with splines on repose time and
tilt recovery. Splines extrapolate as constants, so out-of-range states saturate. Tilt recovery
is capped at the highest value seen at an actual onset (≈1.65×): above that, the only training
hours come from pauses that hadn't ended, which taught the spline that more inflation means a
lower hazard. The hazard is projected forward hour by hour for 12 h with repose advancing and
tilt inflating at the current rate. "What's driving it" lists per-feature log-odds contributions.

**Walk-forward training.** The model is refitted once per calendar month (Hawaii time). The fit
for a month uses only rows whose outcome was known before that month began, and predicts only
that month, so every stored probability is out-of-sample. The first fit covers December 2024 –
June 2025; there are no predictions before July 2025. `service.train()` redoes every monthly fit
and all predictions (well under a second); it runs when a new month begins and when the episode
catalog changes, so a corrected episode time flows into the history while each month still only
sees data from before its cutoff. Two approximations remain: history uses today's best series
(the USGS tilt release replaces the digitized plots about 60 days late), and an episode's
post-eruption trough is measured up to 2 h after it ends.

Skill is scored on those stored predictions, for completed cycles and including the current
pause. It's shown in the UI and at `/api/probability` → `training.metrics`. Dotted lines on the
probability chart mark the monthly refits.

Out-of-range conditions are flagged in the UI. Examples: tilt recovered well past every historical
onset level, a record-long pause, a non-fountaining vent event since the last episode.

## Dashboard controls

A sticky bar above the charts sets the time range for every timeseries (onset probability,
tilt, tilt rate, tremor and earthquakes): the last 1, 7, 30 or 90 days, **All** (since the series
began on Dec 23, 2024), or a custom start and end date in Hawaii time. The browser remembers the
choice. Every episode, plus the Sept 14 non-fountaining vent event, is shaded on all of them.
The tilt and tremor panels draw every source's hourly median (thin) under the stitched series
(bold) and list each source's share of the grid and its mapping. The "Model inputs" table shows
the latest frame row.

## API

* `/api/series/tilt?hours=` and `/api/series/tremor?hours=`: every source plus the stitched
  series, with mappings and grid shares
* `/api/frame?hours=`: feature frame
* `/api/predictions?hours=`: 12 h onset probability per hourly slot, and the monthly refit times
* `/api/probability`: latest value, factors, warnings and out-of-sample skill
* `/api/status`, `/api/notices`, `/api/messages`, `/api/earthquakes?days=`, `/api/episodes`,
  `/api/firms`, `/api/weather`, `/api/health` (per source, grouped)

The series, predictions and earthquake routes also take `start=` and `end=` (unix seconds) in
place of `hours=`/`days=`.

Long ranges are thinned for display by keeping each bucket's min and max. OpenAPI docs are at
`/docs`.

## Tests

`uv run pytest`
