"""Wave-3: the drain's per-row isolation boundary, and what it must cover.

Separate module from `test_hh_store.py` because w3-product-builder is adding
tests there concurrently (team-lead's standing rule on shared test files).

Round-1 finding [10] installed the rule that a bug in one row must never cost
the rest of the backlog. It was enforced around the solve half only; the grading
half sat outside it, so a post-solve error escaped the loop entirely. Found by
w3-product-builder while building `pokerlab-batch` (P4').
"""

from pathlib import Path

import pytest

from pokerlab.hh.persist import UnsolvableSpot, drain_batch_queue, persist_session
from pokerlab.hh.pokerstars import parse_pokerstars
from pokerlab.hh.population import load_population
from pokerlab.hh.report import grade_session
from pokerlab.store import db
from pokerlab.types import Solution

FIXTURES = Path(__file__).parent / "fixtures" / "hh"
AT = "2026-07-19T00:00:00"


def _session(names=("ps_multiway_flop.txt", "ps_ante_hu.txt")):
    parsed, raws = [], []
    for n in names:
        t = (FIXTURES / n).read_text()
        raws.append(t)
        parsed.append(parse_pokerstars(t))
    report = grade_session(parsed, population=load_population())
    conn = db.connect()
    persist_session(conn, parsed, report, graded_at=AT, raw_texts=raws)
    return conn


def _solution_missing_the_heroes_action() -> Solution:
    """A solution whose action space excludes whatever the hero actually did.

    `drills.scoring.score` raises ValueError on an action it has no entry for,
    which is the real shape of the crash: the solver modelled a different node
    than the one the hero was at.
    """
    return Solution(actions={"check": (0.0, 0.6), "jam": (-1.0, 0.4)},
                    range_ctx="test", source="subgame_solver")


def test_a_grading_error_does_not_kill_the_rest_of_the_backlog() -> None:
    """[10] held for the solve half and not the grade half."""
    conn = _session()
    pending = db.pending_batch(conn)
    assert len(pending) >= 2, "fixture must queue more than one row"
    calls = []

    def solver(spot_key: str, d):
        # exactly ONE row produces a solution whose action space cannot grade
        # the hero's actual action; every other row solves and grades fine
        calls.append(d)
        if len(calls) == 1:
            return _solution_missing_the_heroes_action()
        return Solution(actions={d.action_type: (0.0, 1.0)},
                        range_ctx="t", source="subgame_solver")

    result = drain_batch_queue(conn, solver, graded_at=AT)

    # the drain ran to completion instead of raising, and cost exactly one row
    assert len(calls) == len(pending), "the drain stopped early"
    assert result["failed"] == 1
    assert result["done"] == len(pending) - 1
    statuses = [r["status"] for r in db.batch_rows(conn)]
    assert "running" not in statuses, "a crashed row must not be stranded 'running'"


def test_a_row_that_crashed_grading_is_not_left_running() -> None:
    """The status must describe reality even when the grade half raised."""
    conn = _session(("ps_multiway_flop.txt",))
    assert db.pending_batch(conn), "fixture must queue at least one row"

    def solver(spot_key: str, d):
        return _solution_missing_the_heroes_action()

    drain_batch_queue(conn, solver, graded_at=AT)
    assert not [r for r in db.batch_rows(conn) if r["status"] == "running"]


def test_no_partial_grading_survives_a_failed_row() -> None:
    """[E14] atomicity must survive moving the grade inside the boundary."""
    conn = _session(("ps_multiway_flop.txt",))

    def solver(spot_key: str, d):
        return _solution_missing_the_heroes_action()

    drain_batch_queue(conn, solver, graded_at=AT)
    graded = {(g["hand_id"], g["decision_idx"]) for g in db.gradings(conn)}
    failed = [r for r in db.batch_rows(conn) if r["status"] == "failed"]
    for r in failed:
        assert (r["hand_id"], r["decision_idx"]) not in graded, (
            "a failed row must not have left a grading behind")


def test_an_unsolvable_spot_still_closes_terminally() -> None:
    """Moving the boundary must not swallow the [E15] typed signal."""
    conn = _session(("ps_multiway_flop.txt",))

    def solver(spot_key: str, d):
        raise UnsolvableSpot("nope")

    result = drain_batch_queue(conn, solver, graded_at=AT)
    assert result["unsolvable"] >= 1 and result["failed"] == 0
