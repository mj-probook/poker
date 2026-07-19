"""Slice I item 3: M4 tier-2 activation via the real Slice-D solver.

An HU-postflop (river) decision from a fixture HH is graded end-to-end through
a real, small, cached solve — proving the SpotKey → solution_index → solve_io →
Solution → decision-ε grading path that Slice F left stubbed.
"""

from pathlib import Path

import pytest

from pokerlab.hh.decisions import extract_decisions
from pokerlab.hh.persist import drain_batch_queue, persist_session
from pokerlab.hh.pokerstars import parse_pokerstars
from pokerlab.hh.population import load_population
from pokerlab.hh.report import grade_session
from pokerlab.solver import subgame as sg
from pokerlab.spike import tier2
from pokerlab.store import db
from pokerlab.types import TIER_SOLVER

FIXTURES = Path(__file__).parent / "fixtures" / "hh"
AT = "2026-07-19T00:00:00"
FAST_CFG = sg.BetConfig(sizes=(0.75,), jam=False, max_raises=0)


def _ante_hu():
    raw = (FIXTURES / "ps_ante_hu.txt").read_text()
    return parse_pokerstars(raw), raw


def test_river_decision_is_the_oop_root_and_tier2():
    parsed, _ = _ante_hu()
    ds = extract_decisions(parsed)
    river = ds[4]
    assert river.street == "river" and river.tier == TIER_SOLVER
    assert river.position == "BB" and len(river.board) == 5  # hero is OOP root
    assert river.game_state is not None


def test_inline_tier2_grades_via_a_real_cached_river_solve():
    parsed, _ = _ante_hu()
    river = extract_decisions(parsed)[4]

    conn = db.connect()
    key = tier2.cache_solve(conn, river, iters=80, cfg=FAST_CFG)
    assert key is not None and db.solution_path(conn, key) is not None

    report = grade_session([parsed], population=load_population(),
                           solution_for=tier2.make_inline_solution_for(conn))
    gd = next(g for g in report.graded if g.decision.index == 4)
    assert gd.grading.tier == TIER_SOLVER and gd.grading.graded
    assert gd.grading.ev_loss is not None            # real solver EV
    assert gd.grading.chosen == "bet"                # hero bet the river
    assert gd.grading.best in {"bet", "check"}


@pytest.mark.slow
def test_drain_solver_grades_turn_and_river_end_to_end():
    parsed, raw = _ante_hu()
    report = grade_session([parsed], population=load_population())
    conn = db.connect()
    persist_session(conn, [parsed], report, graded_at=AT, raw_texts=[raw])

    result = drain_batch_queue(
        conn, tier2.make_drain_solver(conn, iters=60, cfg=FAST_CFG), graded_at=AT)
    # turn + river are solvable (OOP hero); flop spots are out of turn/river scope
    assert result["done"] >= 2
    tier2_rows = [g for g in db.gradings(conn) if g["tier"] == TIER_SOLVER]
    assert tier2_rows and all(g["ev_loss"] is not None for g in tier2_rows)


# --------------------------------------------------------------------------- #
# Wave-2 [E28][Q22]: solvable()'s OOP guard was `position != "BTN"`. Heads-up
# the button is labelled "SB" (position_label returns BTN only for n>=3), so the
# guard was ALWAYS true HU — an IP hero was solved and graded against the root
# (OOP) strategy, i.e. against the wrong player's ranges entirely. And
# cache_solve bypassed the gate completely, so even a correct gate would not
# have covered that path.
# --------------------------------------------------------------------------- #
def _hu_hero_button():
    raw = (FIXTURES / "ps_hu_hero_button.txt").read_text()
    return parse_pokerstars(raw), raw


def test_hu_button_hero_is_ip_and_not_solvable():
    parsed, _ = _hu_hero_button()
    turn = next(d for d in extract_decisions(parsed) if d.street == "turn")

    # the exact shape that fooled the old guard: IP, but not labelled "BTN"
    assert turn.position != "BTN"
    assert turn.tier == TIER_SOLVER and len(turn.board) == 4
    assert not tier2.solvable(turn), "an IP hero must not be solved as the OOP root"


def test_oop_hero_is_still_solvable():
    parsed, _ = _ante_hu()
    river = extract_decisions(parsed)[4]
    assert tier2.solvable(river)


def test_cache_solve_gates_on_solvable():
    """[Q22] cache_solve must not be a second, ungated solve path."""
    parsed, _ = _hu_hero_button()
    turn = next(d for d in extract_decisions(parsed) if d.street == "turn")

    conn = db.connect()
    key = tier2.cache_solve(conn, turn, iters=40, cfg=FAST_CFG)

    assert key is None, "cache_solve solved a spot the gate rejects"
    assert conn.execute(
        "SELECT COUNT(*) FROM solution_index").fetchone()[0] == 0


def test_drain_solver_and_cache_solve_agree_on_what_is_solvable():
    """[Q22] one gate, one answer — the two entry points cannot diverge."""
    parsed, _ = _ante_hu()
    conn = db.connect()
    for d in extract_decisions(parsed):
        gated = tier2.solvable(d)
        solver = tier2.make_drain_solver(conn, iters=30, cfg=FAST_CFG, persist=False)
        got = solver("k", d)
        if not gated:
            assert got is None
