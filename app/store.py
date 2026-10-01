"""Durable history + model store.

`store/` (tracked in git) is the system of record; `data/volcano.db` (SQLite) is a working copy
that can be rebuilt from it. See docs/DATA_STORE.md for the layout and the reasoning.

    uv run python -m app.store export        # DB -> store/   (merge only: never removes history)
    uv run python -m app.store hydrate       # store/ -> DB   (never overwrites rows the DB has)
    uv run python -m app.store verify        # checksums, key uniqueness, manifest vs files
    uv run python -m app.store status --gaps # coverage per dataset, missing runs in 10-min series
    uv run python -m app.store prune-models  # drop old model bundles, keep every meta.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import db
from .config import DATA_DIR, ROOT, STORE_DIR
from .fetchers.base import tracked

log = logging.getLogger("store")

SCHEMA_VERSION = 1
SQL_TYPES = {"INTEGER": "Int64", "REAL": "float64", "TEXT": "string"}
# live predictions only; hindcasts are recomputable from model + data, so they aren't stored
PREDICTIONS_SQL = (
    "SELECT t, model, version, trained_at, p12, p24, p72, payload FROM probability_archive "
    "UNION ALL "
    "SELECT t, model, version, ? AS trained_at, p12, p24, p72, payload FROM probability_log "
    "WHERE payload NOT LIKE '{\"hindcast\"%'"
)


@dataclass(frozen=True)
class Dataset:
    name: str
    table: str                    # SQLite table the columns/types come from
    key: tuple[str, ...]          # primary key; the last row written wins
    time_col: str | None = None   # set -> partitioned by UTC month of this column
    grid_s: int | None = None     # regular sampling interval, enables coverage/gap reports
    sql: str | None = None        # custom export query (predictions)
    replace_by: str | None = None # snapshot: a DB group (e.g. one plot) replaces the stored group
    hydrate_to: str | None = None # SQLite table to restore into, if different from `table`


DATASETS = [
    Dataset("tilt", "tilt", ("t",), "t", 600),
    Dataset("rsam", "rsam", ("station", "t"), "t", 600),
    Dataset("earthquakes", "earthquakes", ("id",), "t"),
    Dataset("firms", "firms", ("id",), "t"),
    Dataset("notices", "notices", ("notice_id",), "sent_unix"),
    Dataset("status_history", "status_history", ("fetched_at",), "fetched_at"),
    Dataset("predictions", "probability_archive", ("t", "model", "trained_at"), "t",
            sql=PREDICTIONS_SQL, hydrate_to="probability_archive"),
    Dataset("episodes", "episodes", ("label",)),
    Dataset("episode_suggestions", "episode_suggestions", ("notice_id",)),
    Dataset("tilt_plot", "tilt_plot", ("plot", "t"), replace_by="plot"),
]
KV_PREFIXES = ("sens:", "tilt_release_done", "tilt_plot_meta:", "tilt_stitch")


# ---------------------------------------------------------------- helpers

def _dtypes(table: str) -> dict[str, str]:
    conn = db.connect()
    try:
        return {r["name"]: SQL_TYPES.get(r["type"].upper(), "string")
                for r in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


def _frame(rows: list[dict], dtypes: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(dtypes)).astype(dtypes)


def _digest(df: pd.DataFrame) -> str:
    h = hashlib.sha256(",".join(df.columns).encode())
    if len(df):
        h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False, compression="zstd", compression_level=9)
    os.replace(tmp, path)


def _write_json(path: Path, obj) -> bool:
    text = json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n"
    if path.exists() and path.read_text() == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
    return True


def _months(t: pd.Series) -> pd.Series:
    return pd.to_datetime(t, unit="s", utc=True).dt.strftime("%Y-%m")


def _load_manifest(store: Path) -> dict:
    p = store / "MANIFEST.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"schema_version": SCHEMA_VERSION, "datasets": {}}


def _stats(df: pd.DataFrame, ds: Dataset) -> dict:
    out = {"rows": int(len(df))}
    if ds.time_col and len(df):
        out["t_min"], out["t_max"] = int(df[ds.time_col].min()), int(df[ds.time_col].max())
    return out


def _merge(old: pd.DataFrame | None, new: pd.DataFrame, ds: Dataset) -> pd.DataFrame:
    """Union by key; the DB's row wins. Anything only the store has is kept."""
    if old is not None and len(old):
        if ds.replace_by:
            old = old[~old[ds.replace_by].isin(new[ds.replace_by].unique())]
        new = pd.concat([old.reindex(columns=new.columns), new], ignore_index=True)
    return new.drop_duplicates(list(ds.key), keep="last").sort_values(list(ds.key)).reset_index(drop=True)


def _dataset_files(store: Path, ds: Dataset) -> list[Path]:
    d = store / "datasets" / ds.name
    return sorted(d.glob("*.parquet")) if d.exists() else []


def _read_dataset(store: Path, ds: Dataset) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in _dataset_files(store, ds)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _current_trained_at() -> int:
    return int((db.kv_get("model_meta") or {}).get("trained_at") or 0)


# ---------------------------------------------------------------- export (DB -> store)

def export_all(store_dir: Path | None = None) -> dict:
    """Merge the DB into the store. Only partitions whose content changed are rewritten."""
    store = Path(store_dir or STORE_DIR)
    manifest = _load_manifest(store)
    manifest["schema_version"] = SCHEMA_VERSION
    written, report = 0, {}
    for ds in DATASETS:
        dtypes = _dtypes(ds.table)
        if ds.sql:
            rows = db.query(ds.sql, (_current_trained_at(),))
        else:
            rows = db.query(f"SELECT {', '.join(dtypes)} FROM {ds.table}")
        entry = manifest["datasets"].setdefault(ds.name, {"key": list(ds.key), "partitions": {}})
        entry.update(key=list(ds.key), time_col=ds.time_col, grid_s=ds.grid_s)
        if not rows:
            report[ds.name] = 0
            continue
        df = _frame(rows, dtypes)
        # partitions: monthly files when there is a time column, else a single file
        groups = df.groupby(_months(df[ds.time_col])) if ds.time_col else [("all", df)]
        n = 0
        for part_name, part in groups:
            path = store / "datasets" / ds.name / f"{part_name}.parquet"
            old = pd.read_parquet(path) if path.exists() else None
            merged = _merge(old, part, ds)
            digest = _digest(merged)
            prev = entry["partitions"].get(part_name)
            if prev and prev.get("content_sha256") == digest and path.exists():
                continue
            _write_parquet(merged, path)
            entry["partitions"][part_name] = {
                **_stats(merged, ds), "content_sha256": digest,
                "file_sha256": _sha256(path), "bytes": path.stat().st_size,
            }
            n += 1
        written += n
        report[ds.name] = n
        entry["rows"] = sum(p["rows"] for p in entry["partitions"].values())
        spans = [(p["t_min"], p["t_max"]) for p in entry["partitions"].values() if "t_min" in p]
        if spans:
            entry["t_min"], entry["t_max"] = min(s[0] for s in spans), max(s[1] for s in spans)
    kv_changed = _export_kv(store)
    if written or kv_changed:
        manifest["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_json(store / "MANIFEST.json", manifest)
    report["_files_written"] = written
    return report


def _export_kv(store: Path) -> bool:
    path = store / "kv.json"
    kept = json.loads(path.read_text()) if path.exists() else {}
    for r in db.query("SELECT key, value FROM kv"):
        if r["key"].startswith(KV_PREFIXES):
            v = json.loads(r["value"])
            if isinstance(v, list) and isinstance(kept.get(r["key"]), list):
                v = sorted(set(v) | set(kept[r["key"]]))   # e.g. tilt_release_done only grows
            kept[r["key"]] = v
    return _write_json(path, kept)


@tracked("store_export")
def export_job() -> str:
    """Scheduled job: never raises, reports into source_health like the fetchers."""
    return f"{export_all().get('_files_written', 0)} files written"


# ---------------------------------------------------------------- hydrate (store -> DB)

def hydrate(store_dir: Path | None = None, models: bool = True) -> dict:
    """Load the store into SQLite. Existing DB rows are never overwritten (INSERT OR IGNORE)."""
    store = Path(store_dir or STORE_DIR)
    if not (store / "MANIFEST.json").exists():
        return {}
    if models:
        restore_model(store)                  # first: predictions need the restored trained_at
    trained_at = _current_trained_at()
    out = {}
    for ds in DATASETS:
        df = _read_dataset(store, ds)
        if df.empty:
            continue
        table = ds.hydrate_to or ds.table
        cols = [c for c in _dtypes(table) if c in df.columns]
        n = 0
        for lo in range(0, len(df), 50_000):
            chunk = df.iloc[lo:lo + 50_000][cols].astype(object)
            rows = [tuple(None if pd.isna(v) else v for v in r) for r in chunk.itertuples(index=False)]
            with db.tx() as c:
                cur = c.executemany(
                    f"INSERT OR IGNORE INTO {table}({','.join(cols)}) VALUES({','.join('?' * len(cols))})", rows)
                n += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        out[ds.name] = n
    if trained_at:  # predictions made by the restored model are "live" rows again
        with db.tx() as c:
            c.execute("INSERT OR IGNORE INTO probability_log(t,model,version,p12,p24,p72,payload) "
                      "SELECT t,model,version,p12,p24,p72,payload FROM probability_archive WHERE trained_at=?",
                      (trained_at,))
            c.execute("DELETE FROM probability_archive WHERE trained_at=?", (trained_at,))
    kv = store / "kv.json"
    if kv.exists():
        for k, v in json.loads(kv.read_text()).items():
            if db.kv_get(k) is None:
                db.kv_set(k, v)
    return out


def hydrate_if_empty(store_dir: Path | None = None) -> dict:
    """Startup hook: restore history into a fresh DB so nothing has to be re-pulled."""
    probe = ("SELECT (SELECT COUNT(*) FROM tilt) + (SELECT COUNT(*) FROM rsam) + "
             "(SELECT COUNT(*) FROM earthquakes) AS n")
    if db.query_one(probe)["n"] > 0:
        return {}
    out = hydrate(store_dir)
    if out:
        log.info("hydrated from store: %s", out)
    return out


# ---------------------------------------------------------------- models

def _runtime() -> dict:
    import numpy, sklearn
    return {"python": sys.version.split()[0], "numpy": numpy.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__}


def _git_state() -> dict:
    def run(*a):
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, timeout=10).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain", "--", "app"))}
    except Exception:  # noqa: BLE001 - no git / not a checkout
        return {"commit": None, "dirty": None}


def _data_fingerprint() -> dict:
    out = {}
    for ds in DATASETS:
        if ds.time_col and not ds.sql:
            r = db.query_one(f"SELECT COUNT(*) AS n, MAX({ds.time_col}) AS t_max FROM {ds.table}")
            out[ds.name] = {"rows": r["n"], "t_max": r["t_max"]}
    return out


def register_model(bundle: Path, meta: dict, store_dir: Path | None = None, keep: int = 7) -> str:
    """File a freshly trained model bundle under store/models/<id>/ with full provenance."""
    store = Path(store_dir or STORE_DIR)
    sha = _sha256(bundle)
    stamp = datetime.fromtimestamp(meta["trained_at"], timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_id = f"{stamp}-{sha[:8]}"
    d = store / "models" / model_id
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundle, d / "bundle.joblib")
    _write_json(d / "meta.json", {
        **meta, "id": model_id, "bundle_sha256": sha, "bundle_bytes": bundle.stat().st_size,
        "code": _git_state(), "runtime": _runtime(), "data_fingerprint": _data_fingerprint(),
    })
    (store / "models" / "CURRENT").write_text(model_id + "\n")
    prune_models(store, keep)
    return model_id


def list_models(store_dir: Path | None = None) -> list[dict]:
    root = Path(store_dir or STORE_DIR) / "models"
    metas = [json.loads(p.read_text()) for p in sorted(root.glob("*/meta.json"))] if root.exists() else []
    for m in metas:
        m["has_bundle"] = (root / m["id"] / "bundle.joblib").exists()
    return sorted(metas, key=lambda m: m["trained_at"])


def prune_models(store_dir: Path | None = None, keep: int = 7) -> list[str]:
    """Delete bundles older than the newest `keep`, except the first of each ISO week and CURRENT.
    meta.json (metrics, provenance, checksum) is always kept."""
    store = Path(store_dir or STORE_DIR)
    models = list_models(store)
    current = (store / "models" / "CURRENT").read_text().strip() if (store / "models" / "CURRENT").exists() else None
    protect = {m["id"] for m in models[-keep:]} | ({current} if current else set())
    seen_weeks = set()
    for m in models:
        wk = datetime.fromtimestamp(m["trained_at"], timezone.utc).isocalendar()[:2]
        if wk not in seen_weeks:
            seen_weeks.add(wk)
            protect.add(m["id"])
    dropped = []
    for m in models:
        b = store / "models" / m["id"] / "bundle.joblib"
        if m["id"] not in protect and b.exists():
            b.unlink()
            dropped.append(m["id"])
    return dropped


def restore_model(store_dir: Path | None = None) -> bool:
    """Put the store's CURRENT model in place if the DB/data dir has none and it will load here."""
    store = Path(store_dir or STORE_DIR)
    cur = store / "models" / "CURRENT"
    if not cur.exists() or db.kv_get("model_meta"):
        return False
    d = store / "models" / cur.read_text().strip()
    meta_p, bundle = d / "meta.json", d / "bundle.joblib"
    if not (meta_p.exists() and bundle.exists()):
        log.info("store: CURRENT model has no bundle; the app will retrain")
        return False
    meta = json.loads(meta_p.read_text())
    have, want = _runtime(), meta.get("runtime", {})
    if any(have.get(k) != want.get(k) for k in ("scikit-learn", "numpy")):
        log.info("store: model built with %s, running %s; retraining instead", want, have)
        return False
    if _sha256(bundle) != meta["bundle_sha256"]:
        raise ValueError(f"model bundle checksum mismatch: {bundle}")
    dest = DATA_DIR / "models"
    dest.mkdir(exist_ok=True)
    shutil.copy2(bundle, dest / "current.joblib")
    db.kv_set("model_meta", {k: v for k, v in meta.items()
                             if k not in ("id", "bundle_sha256", "bundle_bytes", "code", "runtime", "data_fingerprint")})
    return True


# ---------------------------------------------------------------- verify / status

def verify(store_dir: Path | None = None) -> list[str]:
    """Return a list of problems (empty = healthy)."""
    store = Path(store_dir or STORE_DIR)
    manifest = _load_manifest(store)
    problems = []
    for ds in DATASETS:
        entry = manifest["datasets"].get(ds.name, {"partitions": {}})
        files = {p.stem: p for p in _dataset_files(store, ds)}
        for name in set(entry["partitions"]) - set(files):
            problems.append(f"{ds.name}/{name}: listed in manifest but file missing")
        for name, p in files.items():
            meta = entry["partitions"].get(name)
            if meta is None:
                problems.append(f"{ds.name}/{name}: file not in manifest")
                continue
            if _sha256(p) != meta["file_sha256"]:
                problems.append(f"{ds.name}/{name}: file checksum mismatch")
                continue
            df = pd.read_parquet(p)
            if len(df) != meta["rows"]:
                problems.append(f"{ds.name}/{name}: {len(df)} rows, manifest says {meta['rows']}")
            if df.duplicated(list(ds.key)).any():
                problems.append(f"{ds.name}/{name}: duplicate keys")
            if ds.time_col and len(df) and not (_months(df[ds.time_col]) == name).all():
                problems.append(f"{ds.name}/{name}: rows outside the partition's month")
    for m in list_models(store):
        b = store / "models" / m["id"] / "bundle.joblib"
        if b.exists() and _sha256(b) != m["bundle_sha256"]:
            problems.append(f"models/{m['id']}: bundle checksum mismatch")
    return problems


def _gaps(store: Path, ds: Dataset, min_bins: int = 6) -> list[tuple[int, int]]:
    df = _read_dataset(store, ds)
    if df.empty:
        return []
    out = []
    for _, g in df.groupby("station") if "station" in df else [(None, df)]:
        t = g["t"].sort_values().to_numpy()
        for a, b in zip(t[:-1], t[1:]):
            if (b - a) // ds.grid_s > min_bins:
                out.append((int(a) + ds.grid_s, int(b) - ds.grid_s))
    return sorted(set(out))


def status(store_dir: Path | None = None, gaps: bool = False) -> str:
    store = Path(store_dir or STORE_DIR)
    manifest = _load_manifest(store)
    fmt = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M")  # noqa: E731
    lines = [f"store: {store}  (schema v{manifest.get('schema_version')}, updated {manifest.get('updated_at', 'never')})"]
    for ds in DATASETS:
        e = manifest["datasets"].get(ds.name)
        if not e or not e.get("rows"):
            lines.append(f"  {ds.name:<20} empty")
            continue
        span = f"{fmt(e['t_min'])} .. {fmt(e['t_max'])} UTC" if "t_min" in e else "snapshot"
        cov = ""
        if ds.grid_s and "t_min" in e:
            df = _read_dataset(store, ds)
            n_series = df["station"].nunique() if "station" in df else 1
            cov = f"  coverage {e['rows'] / n_series / ((e['t_max'] - e['t_min']) / ds.grid_s + 1):.1%}"
        lines.append(f"  {ds.name:<20} {e['rows']:>9,} rows  {span}{cov}")
        if gaps and ds.grid_s:
            for a, b in _gaps(store, ds):
                lines.append(f"      gap {fmt(a)} .. {fmt(b)} ({(b - a + ds.grid_s) / 3600:.1f} h missing)")
    models = list_models(store)
    lines.append(f"  models: {len(models)} registered, {sum(m['has_bundle'] for m in models)} with bundles"
                 + (f", newest {models[-1]['id']}" if models else ""))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.store", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export")
    sub.add_parser("hydrate")
    sub.add_parser("verify")
    st = sub.add_parser("status")
    st.add_argument("--gaps", action="store_true")
    pm = sub.add_parser("prune-models")
    pm.add_argument("--keep", type=int, default=7)
    a = ap.parse_args(argv)
    db.init()
    if a.cmd == "export":
        print(json.dumps(export_all(), indent=2))
    elif a.cmd == "hydrate":
        print(json.dumps(hydrate(), indent=2))
    elif a.cmd == "verify":
        problems = verify()
        print("\n".join(problems) if problems else "ok")
        return 1 if problems else 0
    elif a.cmd == "status":
        print(status(gaps=a.gaps))
    elif a.cmd == "prune-models":
        print("dropped bundles:", prune_models(keep=a.keep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
