# History and model store

Goal: never re-pull data you already have, and keep a complete, auditable record of every dataset
and every model the dashboard has used.

## Layers

| Layer | Where | Role | In git |
|---|---|---|---|
| **Store** | `store/` | Durable system of record. Parquet + JSON. | yes |
| **Working DB** | `data/volcano.db` (SQLite) | What the app reads and writes. Rebuilt from the store by `hydrate`. | no |
| **Caches** | `data/tilt_release/`, `data/models/` | Disposable. | no |

The store is what survives: a new laptop, a cloud container, a corrupted DB. SQLite stays the
working copy because the app's queries, upserts and transactions already depend on it.

## Layout

```
store/
  MANIFEST.json            schema_version, per-dataset key/rows/t_min/t_max, per-partition checksums
  kv.json                  small persisted settings (sensor sensitivities, processed tilt releases, tilt-plot meta)
  datasets/
    tilt/2025-01.parquet            10-min UWD tilt (release + stitched plot tail), monthly
    rsam/YYYY-MM.parquet            10-min RSAM per station
    earthquakes/YYYY-MM.parquet     ComCat events
    firms/YYYY-MM.parquet           VIIRS/MODIS hotspots
    notices/YYYY-MM.parquet         HVO notices, full text + html
    status_history/YYYY-MM.parquet  alert level / color code over time
    predictions/YYYY-MM.parquet     every prediction a model made live, with payload (factors, features, warnings)
    episodes/all.parquet            episode catalog (snapshot)
    episode_suggestions/all.parquet
    tilt_plot/all.parquet           digitized plot traces (snapshot, replaced per plot)
  models/
    CURRENT                         id of the model in use
    <UTC-train-time>-<sha8>/
      meta.json                     metrics, feature ranges, training window, git commit + dirty flag,
                                    library versions, row counts per dataset at training time, bundle sha256
      bundle.joblib                 the fitted models (kept for the newest 7 trainings + first of each ISO week)
```

### Rules

* **Time is unix seconds, UTC**, as in the DB. Time-series datasets are partitioned by the UTC month of
  their time column. A closed month never changes again, so git stores it once.
* **Every dataset has a primary key** (`MANIFEST.json → key`). A partition is the union of what is
  already stored and what the DB has; on the same key the DB row wins. **Export never deletes
  anything**, so an emptier DB (new machine, wiped cache) cannot shrink the store.
* **Idempotent and byte-stable:** a partition is rewritten only if its content hash changed. Running
  export twice writes nothing.
* **Hydrate never overwrites** a row the DB already has (`INSERT OR IGNORE`).
* **Typed columns from the SQLite schema** (`INTEGER→Int64`, `REAL→float64`, `TEXT→string`), so schemas
  don't drift between months. Compression is zstd.
* **Not stored on purpose:** hindcasts (recomputable from model + data), `source_health`, `raw_cache`
  (operational), and the 1-minute tilt CSVs (the DB keeps 10-minute means; the USGS releases stay
  downloadable). The raw plot PNGs are also not archived; only their digitized traces are.

### Models

A model is a function of (code, data, library versions), and pickles don't survive library upgrades.
So the store keeps the fitted `bundle.joblib` for convenience, plus a `meta.json` that is enough to
audit or retrain it: git commit, dirty flag, library versions, per-dataset row counts / latest time at
training, metrics, and the bundle checksum. `meta.json` is kept for every training; old bundles are
pruned (`prune-models`) so git doesn't grow by a bundle per day.

On hydrate the `CURRENT` bundle is restored only if its scikit-learn and numpy versions match the
running ones; otherwise the app retrains from the restored data, which is the safe fallback.

Predictions are linked to the model that made them by `(model, version, trained_at)`, so "what did the
dashboard show on 12 Mar, and which model said it?" is answerable from the store alone.

## Commands

```bash
uv run python -m app.store export         # DB -> store/
uv run python -m app.store hydrate        # store/ -> DB (skips rows the DB already has)
uv run python -m app.store verify         # checksums, duplicate keys, manifest vs files (exit 1 on problems)
uv run python -m app.store status --gaps  # coverage per dataset; lists missing runs in the 10-min series
uv run python -m app.store prune-models --keep 7
```

Automatic behaviour:

* **Startup:** if the DB has no tilt, RSAM or earthquakes, the app hydrates from `store/` before
  seeding episodes, so nothing is re-pulled. (Source gaps are then filled by the normal fetchers.)
* **Daily:** a scheduler job runs `export` (`STORE_AUTOEXPORT=0` turns it off).
* **Every training:** the model is filed under `store/models/` with its provenance.

The app **never commits or pushes**. To make a day's data durable:
`git add store && git commit -m "store: $(date -u +%F)" && git push`.

## Size and speed

Measured on synthetic data at realistic volume (22 months of 10-min tilt and RSAM, 30k quakes, 4k
notices, 95k predictions): **5.3 MB** store, export ~7 s, hydrate ~5 s, verify ~3 s, and a daily
export rewrote **1 file** (the current month). Treat the predictions figure as optimistic: the
synthetic payloads repeat, real ones will compress less. Expect tens of MB of git history per year
for the data, plus a bundle per kept model (size not yet measured on real models).

If the store ever grows past what git handles comfortably, the layout is object-store friendly:
every file is immutable once its month closes, and `MANIFEST.json` carries the checksums.
