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
    assert rows[0]["leak_key"] == "SBjam:10bb"
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


def _hand(conn, uid="uid-1"):
    """A real imported_hands row.

    Gradings reference it for real now that `connect` enables foreign keys
    (wave-3 [B2]); before that these tests passed hand_id=1 against an empty
    table, which the declared REFERENCES silently allowed.
    """
    return db.insert_imported_hand(conn, "PokerStars", "raw", "{}", "t", uid)


def test_tier3_grading_with_evloss_is_rejected():
    conn = db.connect(":memory:")
    hid = _hand(conn)
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_grading(conn, hand_id=hid, decision_idx=0, tier=3,
                          chosen="call", best="fold", ev_loss=1.5,
                          leak_key="x|flop|call", graded_at="t")


def test_tier3_grading_without_evloss_is_accepted():
    conn = db.connect(":memory:")
    hid = _hand(conn)
    rid = db.insert_grading(conn, hand_id=hid, decision_idx=0, tier=3,
                            chosen="call", best="fold", ev_loss=None,
                            leak_key="x|flop|call", graded_at="t")
    assert rid is not None


def test_tier1_grading_with_evloss_is_accepted():
    conn = db.connect(":memory:")
    hid = _hand(conn)
    rid = db.insert_grading(conn, hand_id=hid, decision_idx=0, tier=1,
                            chosen="jam", best="jam", ev_loss=0.0,
                            leak_key="SBjam|preflop|jam", graded_at="t")
    assert rid is not None


# --------------------------------------------------------------------------- #
# Wave-3: `CREATE TABLE IF NOT EXISTS` does nothing to a table that already
# exists, so a pokerlab.db created before a schema change silently keeps the old
# shape. For a nullable column that degrades a feature; for the [B3] uniqueness
# rule it means the protection against double-grading a decision is absent on
# exactly the databases that already hold real data.
# --------------------------------------------------------------------------- #
_OLD_SCHEMA = """
CREATE TABLE imported_hands(id INTEGER PRIMARY KEY, site TEXT NOT NULL,
  raw TEXT NOT NULL, parsed_json TEXT NOT NULL, imported_at TEXT NOT NULL,
  hand_uid TEXT, UNIQUE(site, hand_uid));
CREATE TABLE gradings(id INTEGER PRIMARY KEY,
  hand_id INTEGER NOT NULL REFERENCES imported_hands(id),
  decision_idx INTEGER NOT NULL, tier INTEGER NOT NULL, chosen TEXT NOT NULL,
  best TEXT NOT NULL, ev_loss REAL, leak_key TEXT NOT NULL,
  graded_at TEXT NOT NULL);
"""


def _legacy_db(path, *, duplicate=False):
    """A database in the pre-wave-3 shape: no frequency/flags, no B3 uniqueness."""
    c = sqlite3.connect(str(path))
    c.executescript(_OLD_SCHEMA)
    c.execute("INSERT INTO imported_hands VALUES(1,'PokerStars','r','{}','t','u1')")
    if duplicate:
        c.execute("INSERT INTO gradings VALUES(1,1,0,2,'bet','check',0.5,'k','t')")
        c.execute("INSERT INTO gradings VALUES(2,1,0,2,'bet','check',0.5,'k','t')")
    c.commit()
    c.close()
    return path


def test_opening_a_legacy_db_adds_the_missing_columns(tmp_path):
    p = _legacy_db(tmp_path / "old.db")
    conn = db.connect(p)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(gradings)")}
    assert {"frequency", "flags"} <= cols


def test_opening_a_legacy_db_enforces_the_b3_rule(tmp_path):
    """The constraint has to reach OLD databases — that is the whole point."""
    p = _legacy_db(tmp_path / "old.db")
    conn = db.connect(p)
    db.insert_grading(conn, 1, 0, 2, "bet", "check", 0.5, "k", "t")
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_grading(conn, 1, 0, 2, "bet", "check", 0.5, "k", "t")


def test_opening_it_twice_is_idempotent(tmp_path):
    p = _legacy_db(tmp_path / "old.db")
    db.connect(p)
    db.connect(p)  # must not raise on the second pass


def test_a_legacy_db_that_already_double_graded_fails_loudly(tmp_path):
    """Silently skipping would leave the corruption in place, unmentioned.

    If the unique index cannot be created, this database already contains
    duplicate gradings — which double-count into every leak statistic. That is
    worth refusing to open over, with the query to inspect it.
    """
    p = _legacy_db(tmp_path / "dup.db", duplicate=True)
    with pytest.raises(sqlite3.IntegrityError, match="DUPLICATE gradings"):
        db.connect(p)


def test_legacy_drill_attempts_gains_nullable_ev_loss(tmp_path):
    """A drill_attempts created before off-tree actions carried NOT NULL on
    ev_loss; an off-tree attempt (NULL — the chart never priced it) must be
    insertable after connect, with every legacy row surviving the rebuild.
    SQLite cannot drop NOT NULL via ALTER, so this exercises the table rebuild.
    """
    p = tmp_path / "old.db"
    c = sqlite3.connect(str(p))
    c.execute("CREATE TABLE drill_attempts(id INTEGER PRIMARY KEY,"
              " leak_key TEXT NOT NULL, kind TEXT NOT NULL,"
              " chosen TEXT NOT NULL, correct INTEGER NOT NULL,"
              " ev_loss REAL NOT NULL, ts TEXT NOT NULL)")
    c.execute("INSERT INTO drill_attempts(leak_key, kind, chosen, correct,"
              " ev_loss, ts) VALUES ('k', 'jamfold', 'fold', 1, 0.5, 't1')")
    c.commit()
    c.close()
    conn = db.connect(p)
    db.insert_drill_attempt(conn, "k", "jamfold", "limp", correct=False,
                            ev_loss=None, ts="t2")
    rows = [(r["chosen"], r["ev_loss"]) for r in conn.execute(
        "SELECT chosen, ev_loss FROM drill_attempts ORDER BY id")]
    assert rows == [("fold", 0.5), ("limp", None)]
    db.connect(p)  # rebuild must be idempotent — second open is a no-op
