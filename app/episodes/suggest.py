"""Keyword-parse HANS notice text to suggest catalog entries for review, and derive a
simple 'precursory activity reported' signal used by the models.
"""
from __future__ import annotations

import re

from .. import db

KEYWORDS = ["episode", "fountaining", "spattering", "spatter", "gas pistoning", "gas-piston",
            "overflow", "precursory", "dome fountain"]
PRECURSOR_WORDS = re.compile(
    r"\b(overflow\w*|spatter\w*|precursory|dome fountain\w*|gas[- ]pist\w*|lava flows? from the (north|south))\b", re.I)
PAUSED_WORDS = re.compile(r"\b(paused|pause|no (lava )?(overflows|spattering|activity))\b", re.I)
_onset = re.compile(r"episode\s+(\d+)[^.]{0,120}?\b(began|begun|started|has begun|is underway|onset)\b", re.I)
_end = re.compile(r"episode\s+(\d+)[^.]{0,120}?\b(ended|stopped|ceased|paused)\b", re.I)
_precur = re.compile(r"precursory[^.]{0,80}?episode\s+(\d+)|episode\s+(\d+)\s+precursory", re.I)


def classify(text: str) -> list[tuple[int, str, str]]:
    """All (episode_num, phase, snippet) mentions, phase in {end, onset, precursor}."""
    out = []
    for rx, phase in ((_end, "end"), (_onset, "onset"), (_precur, "precursor")):
        for m in rx.finditer(text):
            if phase == "onset" and "precursor" in m.group(0).lower():
                phase_ = "precursor"
            else:
                phase_ = phase
            n = next(g for g in m.groups() if g and g.isdigit())
            s = max(0, m.start() - 40)
            out.append((int(n), phase_, text[s:m.end() + 160]))
    return out


def scan(since_days: int = 30) -> int:
    rows = db.query(
        "SELECT notice_id, sent_unix, text, synopsis FROM notices WHERE sent_unix >= strftime('%s','now') - ? "
        "AND notice_id NOT IN (SELECT notice_id FROM episode_suggestions)", (since_days * 86400,))
    known = {r["num"]: r for r in db.query("SELECT num, end_t FROM episodes WHERE num IS NOT NULL")}
    out = []
    for r in rows:
        body = (r["synopsis"] or "") + "\n" + (r["text"] or "")
        kws = sorted({k for k in KEYWORDS if k in body.lower()})
        if not kws:
            continue
        for num, phase, snippet in classify(body):
            # only suggest things the catalog doesn't know yet
            if num in known and (phase in ("onset", "precursor") or known[num]["end_t"]):
                continue
            out.append((r["notice_id"], r["sent_unix"], ",".join(kws), num, phase,
                        re.sub(r"\s+", " ", snippet)[:400]))
            break
    provisional_update()
    if out:
        with db.tx() as c:
            c.executemany("INSERT OR IGNORE INTO episode_suggestions(notice_id,sent_unix,keywords,episode_num,phase,"
                          "snippet) VALUES(?,?,?,?,?,?)", out)
    return len(out)


def pending() -> list[dict]:
    return db.query("SELECT * FROM episode_suggestions WHERE status='pending' ORDER BY sent_unix DESC LIMIT 50")


def precursor_series() -> list[tuple[int, int]]:
    """[(sent_unix, flag)] for DU/VAN/SR notices: 1 if the summary reports precursory activity."""
    rows = db.query(
        "SELECT sent_unix, synopsis FROM notices WHERE type_cd IN ('DU','VAN','SR','IS') ORDER BY sent_unix")
    out = []
    for r in rows:
        syn = r["synopsis"] or ""
        flag = 1 if PRECURSOR_WORDS.search(syn) and not re.search(r"\bno\b[^.]{0,20}(overflow|spatter)", syn, re.I) else 0
        out.append((r["sent_unix"], flag))
    return out


_when = re.compile(
    r"\b(began|begun|started|start|ended|stopped|ceased)\b[^.]{0,60}?\bat\s+(?:approximately\s+|about\s+|around\s+)?"
    r"(\d{1,2})(?::(\d{2}))?\s*(a\.?\s?m\.?|p\.?\s?m\.?)?\s*HST"
    r"(?:\s+on)?(?:\s+\w+day,)?\s+([A-Z][a-z]+\s+\d{1,2})(?:,\s*(\d{4}))?", re.I)


def sentences(text: str) -> list[str]:
    """Split into sentences after collapsing 'a.m.'/'p.m.' so their periods don't end sentences."""
    t = re.sub(r"\b([ap])\.\s?m\.", r"\1m", text, flags=re.I)
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", t) if x.strip()]


def parse_event_time(sentence: str, sent_unix: int) -> tuple[str, int] | None:
    """('onset'|'end', unix) from e.g. 'began at 10:30 a.m. HST on August 25, 2026'."""
    from datetime import datetime

    from ..config import HST

    m = _when.search(sentence)
    if not m:
        return None
    verb, hh, mm, ap, md, year = m.groups()
    h = int(hh)
    if ap:
        h = h % 12 + (12 if ap.lower().startswith("p") else 0)
    elif h > 23:
        return None
    y = int(year) if year else datetime.fromtimestamp(sent_unix, HST).year
    try:
        dt = datetime.strptime(f"{md} {y}", "%B %d %Y").replace(hour=h, minute=int(mm or 0), tzinfo=HST)
    except ValueError:
        return None
    t = int(dt.timestamp())
    if not (sent_unix - 4 * 86400 <= t <= sent_unix + 3600):
        return None
    # "began and ended ... at 7:54" -> the verb nearest the time wins
    between = sentence[m.start():m.start() + len(m.group(0))]
    ends = [x.start() for x in re.finditer(r"\b(ended|stopped|ceased)\b", between, re.I)]
    starts = [x.start() for x in re.finditer(r"\b(began|begun|started|start)\b", between, re.I)]
    phase = "end" if ends and (not starts or max(ends) > max(starts)) else "onset"
    return phase, t


def provisional_update(nxt: int | None = None, since_unix: int | None = None, dry: bool = False):
    """Add/close the next episode from HVO notices before the USGS table catches up.

    Only episode (last catalogued number + 1) is touched, and the USGS table overwrites it later.
    """
    if nxt is None:
        last = db.query_one("SELECT MAX(num) AS n FROM episodes WHERE kind='fountaining' AND source!='hans_provisional'")
        nxt = (last["n"] or 0) + 1
    import time as _time

    since = since_unix if since_unix is not None else int(_time.time()) - 30 * 86400
    rows = db.query("SELECT notice_id, sent_unix, synopsis, text FROM notices WHERE sent_unix >= ? "
                    "AND type_cd IN ('VAN','VV','SR','DU','IS') ORDER BY sent_unix", (since,))
    start = end = None
    rx = re.compile(rf"\bepisode\s+{nxt}\b", re.I)
    for r in rows:
        body = (r["synopsis"] or "") + " " + (r["text"] or "")
        for sent in sentences(body):
            if not rx.search(sent) or re.search(r"precursor|likely|expected|forecast|possib|may |could ", sent, re.I):
                continue
            got = parse_event_time(sent, r["sent_unix"])
            if not got:
                continue
            phase, t = got
            if phase == "onset" and start is None:
                start = t
            elif phase == "end" and end is None:
                end = t
    if end is not None and start is not None and end <= start:
        end = None
    if dry:
        return start, end
    if start is None:
        return 0

    with db.tx() as c:
        c.execute(
            """INSERT INTO episodes(label,num,kind,start_t,end_t,notes,source,updated_at)
               VALUES(?,?,'fountaining',?,?,?,'hans_provisional',?)
               ON CONFLICT(label) DO UPDATE SET start_t=excluded.start_t, end_t=COALESCE(excluded.end_t, episodes.end_t),
                 updated_at=excluded.updated_at WHERE episodes.source='hans_provisional'""",
            (str(nxt), nxt, start, end, "Provisional, parsed from HVO notices; replaced when the USGS table updates.",
             int(_time.time())))
    return 1
