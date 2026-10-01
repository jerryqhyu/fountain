"""History + model store: round trip, idempotence, monotonic merge, tamper detection, model registry."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from app import db, store


def ts(y, m, d, h=0):
    return int(datetime(y, m, d, h, tzinfo=timezone.utc).timestamp())


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "a.db")
    monkeypatch.setattr(store, "DATA_DIR", tmp_path / "data")
    (tmp_path / "data").mkdir()
    db.init()
    return tmp_path


def fresh_db(env, name="b.db"):
    db.DB_PATH = env / name
    db.init()


def seed(live_trained_at=1000):
    with db.tx() as c:
        c.executemany("INSERT INTO tilt(t,v,src) VALUES(?,?,?)", [
            (ts(2024, 12, 30, 0) + 600 * i, 1.0 + i, "release:x.csv") for i in range(5)] + [
            (ts(2025, 1, 5, 0), 2.5, "plot:2day"), (ts(2025, 2, 9, 12), None, "plot:week")])
        c.executemany("INSERT INTO rsam(station,t,counts,ums) VALUES(?,?,?,?)",
                      [("UWE", ts(2025, 1, 5) + 600 * i, 100.0 + i, 0.5) for i in range(3)])
        c.execute("INSERT INTO earthquakes(id,t,lat,lon,depth,mag,magtype,place,region) "
                  "VALUES('hv1',?,19.4,-155.3,1.2,2.1,'ml','5 km S of x','summit')", (ts(2025, 1, 6),))
        c.execute("INSERT INTO notices(notice_id,sent_unix,type_cd,type_title,title,synopsis,text,html,url) "
                  "VALUES('n1',?,'DU','Daily','Update','syn','body ī text','<p>x</p>','http://u')", (ts(2025, 2, 1),))
        c.execute("INSERT INTO firms(id,t,lat,lon,frp,bright,satellite,confidence,source) "
                  "VALUES('f1',?,19.41,-155.28,12.5,330.1,'N','h','VIIRS_SNPP_NRT')", (ts(2025, 2, 2),))
        c.execute("INSERT INTO status_history(fetched_at,alert_level,color_code,alert_date,notice_id,synopsis) "
                  "VALUES(?,'WATCH','ORANGE','d','n1','s')", (ts(2025, 2, 2),))
        c.execute("INSERT INTO episodes(label,num,kind,start_t,end_t,notes,source,updated_at) "
                  "VALUES('55',55,'fountaining',?,NULL,'open','usgs_table',1)", (ts(2025, 2, 3),))
        c.execute("INSERT INTO episode_suggestions(notice_id,sent_unix,keywords,episode_num,phase,snippet) "
                  "VALUES('n1',?,'k',55,'onset','s')", (ts(2025, 2, 3),))
        c.execute("INSERT INTO tilt_plot(plot,t,v,vmin,vmax,fetched_at) VALUES('2day',?,1.0,0.9,1.1,5)", (ts(2025, 2, 3),))
        # live prediction by the current model, one hindcast (must not be stored), one archived by an old model
        c.execute("INSERT INTO probability_log(t,model,version,p12,p24,p72,payload) VALUES(?,'hazard','h1',.1,.2,.3,'{\"p\":1}')", (ts(2025, 2, 4),))
        c.execute("INSERT INTO probability_log(t,model,version,p12,p24,p72,payload) VALUES(?,'hazard','h1',.1,.2,.3,'{\"hindcast\": true}')", (ts(2025, 2, 3),))
        c.execute("INSERT INTO probability_archive(t,model,version,trained_at,p12,p24,p72,payload) VALUES(?,'ml','m0',500,.4,.5,.6,'{}')", (ts(2025, 1, 30),))
    db.kv_set("model_meta", {"trained_at": live_trained_at})
    db.kv_set("sens:HV.UWE..HHZ", 1.2e9)
    db.kv_set("tilt_release_done", ["a.zip"])
    db.kv_set("not_persisted", 1)


def dump(sql):
    return db.query(sql)


def test_round_trip_and_partitions(env):
    seed()
    s = env / "store"
    rep = store.export_all(s)
    files = sorted(p.relative_to(s).as_posix() for p in (s / "datasets").rglob("*.parquet"))
    assert "datasets/tilt/2024-12.parquet" in files and "datasets/tilt/2025-01.parquet" in files
    assert "datasets/tilt/2025-02.parquet" in files and "datasets/episodes/all.parquet" in files
    assert rep["_files_written"] == len(files)
    assert store.verify(s) == []

    snapshot = {t: dump(f"SELECT * FROM {t} ORDER BY 1,2") for t in
                ("tilt", "rsam", "earthquakes", "notices", "firms", "status_history", "episodes", "episode_suggestions", "tilt_plot")}
    fresh_db(env)
    store.hydrate(s, models=False)
    for t, rows in snapshot.items():
        assert dump(f"SELECT * FROM {t} ORDER BY 1,2") == rows, t
    assert db.kv_get("sens:HV.UWE..HHZ") == 1.2e9 and db.kv_get("tilt_release_done") == ["a.zip"]
    assert db.kv_get("not_persisted") is None
    # hindcast never stored; live + archived predictions both come back (as archive: no model restored)
    pred = dump("SELECT t, model, trained_at FROM probability_archive ORDER BY t")
    assert [(p["model"], p["trained_at"]) for p in pred] == [("ml", 500), ("hazard", 1000)]
    assert dump("SELECT COUNT(*) AS n FROM probability_log")[0]["n"] == 0


def test_export_is_idempotent_and_byte_stable(env):
    seed()
    s = env / "store"
    store.export_all(s)
    before = {p: p.read_bytes() for p in s.rglob("*") if p.is_file()}
    rep = store.export_all(s)
    assert rep["_files_written"] == 0
    assert {p: p.read_bytes() for p in s.rglob("*") if p.is_file()} == before


def test_export_never_shrinks_history_and_db_wins_conflicts(env):
    seed()
    s = env / "store"
    store.export_all(s)
    fresh_db(env)                                     # an emptier DB (e.g. a new machine)
    with db.tx() as c:
        c.execute("INSERT INTO tilt(t,v,src) VALUES(?,?,?)", (ts(2024, 12, 30, 0), 99.0, "release:y.csv"))
    store.export_all(s)
    rows = store._read_dataset(s, next(d for d in store.DATASETS if d.name == "tilt"))
    assert len(rows) == 7                             # nothing lost
    assert rows.loc[rows.t == ts(2024, 12, 30, 0), "v"].iloc[0] == 99.0   # DB wins on the same key
    assert db.kv_get("tilt_release_done") is None
    assert json.loads((s / "kv.json").read_text())["tilt_release_done"] == ["a.zip"]


def test_hydrate_does_not_overwrite_db_rows(env):
    seed()
    s = env / "store"
    store.export_all(s)
    with db.tx() as c:
        c.execute("UPDATE tilt SET v=-1 WHERE t=?", (ts(2024, 12, 30, 0),))
    store.hydrate(s, models=False)
    assert dump(f"SELECT v FROM tilt WHERE t={ts(2024, 12, 30, 0)}")[0]["v"] == -1


def test_verify_catches_corruption_and_stray_files(env):
    seed()
    s = env / "store"
    store.export_all(s)
    p = s / "datasets" / "rsam" / "2025-01.parquet"
    p.write_bytes(p.read_bytes()[:-8] + b"corrupt!")
    (s / "datasets" / "tilt" / "2030-01.parquet").write_bytes(b"x")
    problems = " | ".join(store.verify(s))
    assert "rsam/2025-01" in problems and "tilt/2030-01: file not in manifest" in problems


def test_status_and_gaps(env):
    seed()
    s = env / "store"
    store.export_all(s)
    with db.tx() as c:
        c.execute("INSERT INTO rsam(station,t,counts,ums) VALUES('UWE',?,1,1)", (ts(2025, 1, 5) + 600 * 100,))
    store.export_all(s)
    out = store.status(s, gaps=True)
    assert "rsam" in out and "gap 2025-01-05 00:30 .. 2025-01-05 16:30 (16.2 h missing)" in out


def fake_model(env, trained_at, name):
    b = env / name
    b.write_bytes(name.encode() * 10)
    return b, {"trained_at": trained_at, "n_rows": 5, "metrics": {"x": 1}, "versions": {"hazard": "h"}}


def test_model_registry_prune_and_restore(env, monkeypatch):
    s = env / "store"
    ids = []
    for i in range(12):                              # 12 daily trainings across two ISO weeks
        b, meta = fake_model(env, ts(2025, 3, 3 + i), f"m{i}")
        ids.append(store.register_model(b, meta, s, keep=3))
    models = store.list_models(s)
    assert len(models) == 12                         # every training keeps its meta.json
    kept = [m["id"] for m in models if m["has_bundle"]]
    assert set(ids[-3:]) <= set(kept)                # newest three
    assert ids[0] in kept and ids[7] in kept         # first of each ISO week (Mon 3 Mar, Mon 10 Mar)
    assert len(kept) == 5
    m = json.loads((s / "models" / ids[-1] / "meta.json").read_text())
    assert m["bundle_sha256"] and m["runtime"]["scikit-learn"] and "code" in m and "data_fingerprint" in m

    # restore into a fresh DB: bundle copied, meta restored, runtime mismatch refuses
    fresh_db(env)
    assert store.restore_model(s) is True
    assert (env / "data" / "models" / "current.joblib").read_bytes() == b"m11" * 10
    assert db.kv_get("model_meta")["trained_at"] == ts(2025, 3, 14)
    fresh_db(env, "c.db")
    monkeypatch.setattr(store, "_runtime", lambda: {"scikit-learn": "0.0", "numpy": "0.0"})
    assert store.restore_model(s) is False


def test_live_predictions_return_to_probability_log_with_their_model(env):
    seed(live_trained_at=1000)
    s = env / "store"
    b, meta = fake_model(env, 1000, "mm")
    store.register_model(b, meta, s)
    store.export_all(s)
    fresh_db(env)
    store.hydrate(s)
    assert [r["model"] for r in dump("SELECT model FROM probability_log")] == ["hazard"]
    assert [r["model"] for r in dump("SELECT model FROM probability_archive")] == ["ml"]
