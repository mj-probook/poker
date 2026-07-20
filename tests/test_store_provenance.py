"""[R4-1]: the tier-2 range approximation must reach the leak report.

PLAN §5.3 promises the approximation is "recorded in each Solution's provenance
(`range_ctx`) AND in the leak report". The first half held — every tier-2
Solution carries `ranges=uniform|stack=symmetric`. The second half did not: no
column stored it, so it was dropped at persist and unrecoverable downstream, and
a tier-2 grade reached the user bucketed as plain "exact — solver-graded".

That is the disclosure the plan's own honesty note exists to make. A number
derived from both players holding uniform ranges is exact *for that game*, not
for the hand as played, and the user cannot tell the difference from an EV-loss
figure alone.

Store side only — the render half is the builder's.
"""

from pathlib import Path

import pytest

from pokerlab.hh.grade import grade_tier2
from pokerlab.spike.tier2 import RANGE_PROVENANCE
from pokerlab.store import db, views
from pokerlab.types import Solution

AT = "2026-07-20T00:00:00"


def _hand(conn, uid="u1"):
    return db.insert_imported_hand(conn, "PokerStars", "raw", "{}", AT, uid)


def _solution(ctx=RANGE_PROVENANCE):
    return Solution(actions={"bet": (0.0, 0.7), "check": (-1.5, 0.3)},
                    range_ctx=ctx, source="subgame_solver")


def test_a_tier2_grading_carries_its_solutions_provenance():
    """The grade path must copy it off the Solution, or nothing downstream can."""
    class _D:
        index, formation, street, action_type = 0, "SRP", "flop", "bet"
        pot_bb = 10.0
    g = grade_tier2(_D(), _solution())
    assert g.provenance == RANGE_PROVENANCE


def test_provenance_survives_the_write():
    conn = db.connect()
    hid = _hand(conn)
    db.insert_grading(conn, hid, 0, 2, "bet", "bet", 0.5, "SRP|flop|bet|-", AT,
                      provenance=RANGE_PROVENANCE)
    assert db.gradings(conn)[0]["provenance"] == RANGE_PROVENANCE


def test_the_leak_report_discloses_how_many_decisions_were_approximated():
    """The half PLAN §5.3 promised and the store could not deliver."""
    conn = db.connect()
    hid = _hand(conn)
    # one leak_key, three decisions: two graded against an approximated
    # tier-2 reference, one against an exact tier-1 chart
    for i, prov in enumerate((RANGE_PROVENANCE, RANGE_PROVENANCE, None)):
        db.insert_grading(conn, hid, i, 2 if prov else 1, "bet", "bet", 0.4,
                          "SRP|flop|bet|-", AT, provenance=prov)

    row = views.hh_leak_report(conn)[0]
    assert row["decisions"] == 3
    assert row["approx_decisions"] == 2, "must count, not flag"
    assert row["provenance"] == RANGE_PROVENANCE


def test_an_all_exact_leak_reports_no_approximation():
    """Absence must be visible as absence, not as a missing key."""
    conn = db.connect()
    hid = _hand(conn)
    db.insert_grading(conn, hid, 0, 1, "jam", "jam", 0.2, "SB|preflop|jam|10", AT)

    row = views.hh_leak_report(conn)[0]
    assert row["approx_decisions"] == 0
    assert row["provenance"] is None


def test_a_legacy_db_gains_the_provenance_column(tmp_path):
    """The COLUMN case: a table that already exists never gains one on its own.

    Same shape as the [P3'] frequency/flags ALTER — `CREATE TABLE IF NOT EXISTS`
    is a no-op on an existing table, so without this a database holding real
    gradings would silently keep the old shape and every write would raise.
    """
    import sqlite3

    p = tmp_path / "old.db"
    c = sqlite3.connect(str(p))
    c.executescript("""
        CREATE TABLE imported_hands(id INTEGER PRIMARY KEY, site TEXT NOT NULL,
          raw TEXT NOT NULL, parsed_json TEXT NOT NULL, imported_at TEXT NOT NULL,
          hand_uid TEXT, UNIQUE(site, hand_uid));
        CREATE TABLE gradings(id INTEGER PRIMARY KEY, hand_id INTEGER NOT NULL,
          decision_idx INTEGER NOT NULL, tier INTEGER NOT NULL, chosen TEXT NOT NULL,
          best TEXT NOT NULL, ev_loss REAL, leak_key TEXT NOT NULL,
          graded_at TEXT NOT NULL);
    """)
    c.commit()
    c.close()

    conn = db.connect(p)                       # must not raise
    hid = _hand(conn)
    db.insert_grading(conn, hid, 0, 2, "bet", "bet", 0.5, "SRP|flop|bet|-", AT,
                      provenance=RANGE_PROVENANCE)
    assert db.gradings(conn)[0]["provenance"] == RANGE_PROVENANCE


@pytest.mark.parametrize("tier", [1, 3])
def test_tiers_without_a_reference_solution_store_no_provenance(tier):
    """None must mean "no approximation to disclose", never an empty string.

    Inventing a provenance for a chart grade would make an absent disclosure
    look like a present one, which is the failure this whole item is about.
    """
    conn = db.connect()
    hid = _hand(conn)
    db.insert_grading(conn, hid, 0, tier, "jam", "" if tier == 3 else "jam",
                      None if tier == 3 else 0.1, "SB|preflop|jam|10", AT)
    assert db.gradings(conn)[0]["provenance"] is None
