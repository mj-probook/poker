"""Slice E — sqlite wrapper around store/schema.sql (impl doc §2; plan §3).

The wrapper only applies the checked-in schema and offers typed insert/query
helpers; derived metrics live in views.py, never as tables. The load-bearing
grading-honesty invariant (tier-3 rows may never carry ev_loss) is enforced by a
schema trigger — exercised directly here.
"""

import sqlite3

import pytest

from pokerlab.store import db


def test_connect_applies_full_schema(tmp_path):
    conn = db.connect(tmp_path / "lab.db")
    names = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"drill_attempts", "sr_state", "gradings"} <= names


def test_insert_and_read_drill_attempt():
    conn = db.connect(":memory:")
    rid = db.insert_drill_attempt(conn, "SBjam:10bb", "jamfold", "jam",
                                  correct=True, ev_loss=0.0, ts="2026-07-19T00:00:00")
    rows = db.drill_attempts(conn)
    assert len(rows) == 1
    assert rows[0]["id"] == rid
    assert rows[0]["spot_key"] == "SBjam:10bb"
    assert rows[0]["correct"] == 1


def test_upsert_sr_state_inserts_then_updates():
    conn = db.connect(":memory:")
    db.upsert_sr_state(conn, "SBjam:10bb", easiness=2.5, interval_days=1.0,
                       reps=1, due="2026-07-20T00:00:00")
    got = db.get_sr_state(conn, "SBjam:10bb")
    assert got["reps"] == 1 and got["easiness"] == pytest.approx(2.5)
    # second write to the same leak_key updates in place (PRIMARY KEY conflict).
    db.upsert_sr_state(conn, "SBjam:10bb", easiness=2.6, interval_days=6.0,
                       reps=2, due="2026-07-26T00:00:00")
    got = db.get_sr_state(conn, "SBjam:10bb")
    assert got["reps"] == 2 and got["interval_days"] == pytest.approx(6.0)
    assert len(db.all_sr_state(conn)) == 1   # updated, not duplicated


def test_get_sr_state_missing_returns_none():
    conn = db.connect(":memory:")
    assert db.get_sr_state(conn, "nope") is None


def test_tier3_grading_with_evloss_is_rejected():
    conn = db.connect(":memory:")
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_grading(conn, hand_id=1, decision_idx=0, tier=3,
                          chosen="call", best="fold", ev_loss=1.5,
                          leak_key="x|flop|call", graded_at="t")


def test_tier3_grading_without_evloss_is_accepted():
    conn = db.connect(":memory:")
    rid = db.insert_grading(conn, hand_id=1, decision_idx=0, tier=3,
                            chosen="call", best="fold", ev_loss=None,
                            leak_key="x|flop|call", graded_at="t")
    assert rid is not None


def test_tier1_grading_with_evloss_is_accepted():
    conn = db.connect(":memory:")
    rid = db.insert_grading(conn, hand_id=1, decision_idx=0, tier=1,
                            chosen="jam", best="jam", ev_loss=0.0,
                            leak_key="SBjam|preflop|jam", graded_at="t")
    assert rid is not None
