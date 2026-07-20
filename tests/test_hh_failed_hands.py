"""Wave-3 sweep [R1]: a failed hand must leave a DURABLE trace.

PLAN §8 M4 promises hands failing parse/replay are "isolated in `failed_hands`
and surfaced in the session summary — never silently dropped". Round-1 [8]
delivered the isolation as an in-memory dataclass on `SessionReport`, which is
correct for the CLI and vanishes with the process. On the intended nightly-cron
path nobody reads stdout, so by morning a failed hand had left no trace at all:
`/api/report` showed the session as complete and short by one hand, with no
failure count anywhere.

This covers the STORE half. The report surface is w3-product-builder's.
"""

import sqlite3
from pathlib import Path

import pytest

from pokerlab.hh.persist import failed_hand_report, persist_session
from pokerlab.hh.pokerstars import parse_pokerstars
from pokerlab.hh.population import load_population
from pokerlab.hh.report import FailedHand, grade_session
from pokerlab.store import db

FIXTURES = Path(__file__).parent / "fixtures" / "hh"
AT = "2026-07-19T00:00:00"
LATER = "2026-07-20T00:00:00"


def _pair():
    """Two DISTINCT hands — same-hand-twice would dedup and mask the result."""
    out = []
    # good hand first. It must be one whose decisions actually grade -- the
    # preflop fold parses fine but its only decision is graded=False, so using
    # it as the good hand would assert nothing about gradings surviving.
    for name in ("ps_multiway_flop.txt", "ps_preflop_fold.txt"):
        raw = (FIXTURES / name).read_text()
        out.append((raw, parse_pokerstars(raw)))
    return out


def _session_with_one_broken_hand(conn, *, graded_at=AT, break_it=True):
    """r3-product's acceptance shape: one good hand, one that cannot be graded.

    The break is injected at the report level rather than by shipping a corrupt
    fixture, so the test names exactly which hand failed and why, without
    depending on some particular parser bug staying unfixed to stay meaningful.
    """
    pair = _pair()
    hands = [p for _, p in pair]
    raws = [pair[0][0], pair[1][0]]
    report = grade_session(hands, population=load_population())
    if break_it:
        report.graded[:] = [g for g in report.graded if g.hand_index != 1]
        report.failed_hands.append(
            FailedHand(1, hands[1].hand_id, "ValueError: unparseable street"))
    return persist_session(conn, hands, report, graded_at=graded_at,
                           raw_texts=raws)


def test_a_failed_hand_is_recorded_not_dropped():
    conn = db.connect()
    counts = _session_with_one_broken_hand(conn)

    assert counts["failed"] == 1
    rows = failed_hand_report(conn)
    assert len(rows) == 1
    assert rows[0]["reason"] == "ValueError: unparseable street"
    assert rows[0]["imported_at"] == AT


def test_the_good_hand_still_lands():
    """One bad hand must not cost the session — the [8] invariant, persisted."""
    conn = db.connect()
    counts = _session_with_one_broken_hand(conn)

    assert counts["imported"] == 1 and counts["graded"] > 0
    assert len(db.gradings(conn)) == counts["graded"]


def test_the_raw_text_is_kept_so_the_hand_can_be_reimported():
    """Storing the text IS the re-import path; without it the record is a stub."""
    conn = db.connect()
    _session_with_one_broken_hand(conn)
    expected = (FIXTURES / "ps_preflop_fold.txt").read_text()
    assert failed_hand_report(conn)[0]["raw"] == expected


def test_a_failed_hand_does_not_claim_its_dedup_uid():
    """[E16]/[E17] must survive: recording the failure must not make it permanent.

    If the failed hand claimed (site, hand_uid) in `imported_hands`, a later
    re-import after the parser fix would be skipped as a duplicate and the hand
    would be lost forever — the failure record would have caused the loss it
    exists to prevent.
    """
    conn = db.connect()
    _session_with_one_broken_hand(conn)

    uid = failed_hand_report(conn)[0]["hand_uid"]
    assert uid, "the parse got far enough to know the hand number"
    assert not conn.execute(
        "SELECT 1 FROM imported_hands WHERE hand_uid=?", (uid,)).fetchone()


def test_reimporting_after_a_fix_clears_the_failure():
    """Otherwise the summary reports a hand that now grades fine, forever."""
    conn = db.connect()
    _session_with_one_broken_hand(conn)
    assert len(failed_hand_report(conn)) == 1

    # the parser is fixed: the same session re-imports, nothing fails
    _session_with_one_broken_hand(conn, graded_at=LATER, break_it=False)

    assert failed_hand_report(conn) == [], (
        "a fixed hand must stop being reported as failed")


def test_failures_are_scoped_to_an_import_batch():
    conn = db.connect()
    _session_with_one_broken_hand(conn)
    assert len(failed_hand_report(conn, AT)) == 1
    assert failed_hand_report(conn, LATER) == []


def test_the_record_is_written_in_the_import_transaction():
    """A rollback must take the failure record with it.

    `persist_session` is one transaction so a session lands whole or not at all
    (round-1 [11]). A failure row committed outside it would survive a rolled-
    back import and report a failure for a session that never landed.
    """
    conn = db.connect()
    pair = _pair()
    hands = [p for _, p in pair]
    report = grade_session(hands, population=load_population())
    report.graded[:] = [g for g in report.graded if g.hand_index != 0]
    report.failed_hands.append(FailedHand(0, hands[0].hand_id, "boom"))

    # hand 0 is written to failed_hands, then hand 1 raises before the commit
    with pytest.raises(IndexError):
        persist_session(conn, hands, report, graded_at=AT,
                        raw_texts=[pair[0][0]])   # one raw for two hands

    assert failed_hand_report(conn) == [], "the failure row outlived its rollback"


# --------------------------------------------------------------------------- #
# A database created before this table existed must gain it on connect.
# --------------------------------------------------------------------------- #
def test_a_legacy_db_gains_the_failed_hands_table(tmp_path):
    p = tmp_path / "old.db"
    c = sqlite3.connect(str(p))
    c.executescript("""
        CREATE TABLE imported_hands(id INTEGER PRIMARY KEY, site TEXT NOT NULL,
          raw TEXT NOT NULL, parsed_json TEXT NOT NULL, imported_at TEXT NOT NULL,
          hand_uid TEXT, UNIQUE(site, hand_uid));
    """)
    c.commit()
    c.close()

    conn = db.connect(p)
    db.insert_failed_hand(conn, "PokerStars", "raw", "why", AT)
    assert len(db.failed_hands(conn)) == 1
