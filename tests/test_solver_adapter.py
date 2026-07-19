"""Slice D — Solution adapter + solve writer + solution_index (impl §3 Slice D)."""

from __future__ import annotations

import numpy as np

from pokerlab.engine.cards import card_from_str
from pokerlab.solver import subgame as sg
from pokerlab.solver.adapter import root_solution_by_class
from pokerlab.solver.solve_io import load_solve, write_solve
from pokerlab.store import db
from pokerlab.types import Solution


def _cards(s: str) -> tuple[int, ...]:
    return tuple(card_from_str(s[i:i + 2]) for i in range(0, len(s), 2))


def _range(combos: list[str]) -> np.ndarray:
    r = np.zeros(sg.NUM_COMBOS)
    for cs in combos:
        a, b = _cards(cs)
        r[sg.COMBO_INDEX[(min(a, b), max(a, b))]] = 1.0
    return r


def _solved():
    board = _cards("AsKd7h2c9s")
    r0 = _range(["QhQs", "JdJc", "TdTh"])  # OOP
    r1 = _range(["8d8h", "6c6d", "AcAh"])  # IP
    cfg = sg.BetConfig(sizes=(0.75,), jam=False, max_raises=0)
    tree = sg.build_tree(board, pot0=10.0, stack=20.0, cfg=cfg)
    s = sg.SubgameSolver(tree, board, r0, r1, pot0=10.0)
    s.iterate(200)
    return s


def test_adapter_emits_solution_per_class_in_range():
    s = _solved()
    sols = root_solution_by_class(s, player=0, range_ctx="ctx")
    # OOP range is QQ, JJ, TT -> exactly those classes appear
    assert set(sols) == {"QQ", "JJ", "TT"}
    for sol in sols.values():
        assert isinstance(sol, Solution)
        assert sol.source == "subgame_solver"
        assert sol.range_ctx == "ctx"


def test_adapter_frequencies_sum_to_one_and_evs_finite():
    s = _solved()
    sols = root_solution_by_class(s, player=0)
    root_labels = s.decisions[0].actions
    for sol in sols.values():
        assert set(sol.actions) == set(root_labels)
        freq_sum = sum(f for _, f in sol.actions.values())
        assert freq_sum == np.float64(freq_sum)  # numeric
        assert abs(freq_sum - 1.0) < 1e-6
        for ev, _ in sol.actions.values():
            assert np.isfinite(ev)


def test_write_solve_and_load_roundtrip(tmp_path):
    s = _solved()
    path, expl = write_solve(s, spot_key="AsKd7h:rainbow_dry|40|BB", path=tmp_path / "solve.json")
    data = load_solve(path)
    assert data["source"] == "subgame_solver"
    assert data["solver_version"] == sg.SOLVER_VERSION
    assert abs(data["exploitability_bb"] - expl) < 1e-12
    assert set(data["solutions"]) == {"QQ", "JJ", "TT"}
    # a Solution reconstitutes from the file
    qq = data["solutions"]["QQ"]
    assert qq["source"] == "subgame_solver"
    assert abs(sum(f for _, f in qq["actions"].values()) - 1.0) < 1e-6


def test_solution_index_insert_is_queryable():
    conn = db.connect(":memory:")
    db.index_solution(conn, "spotA", "solves/spotA.json", sg.SOLVER_VERSION, 0.004)
    row = conn.execute("SELECT * FROM solution_index WHERE spot_key='spotA'").fetchone()
    assert row["path"] == "solves/spotA.json"
    assert row["solver_version"] == sg.SOLVER_VERSION
    assert abs(row["exploitability"] - 0.004) < 1e-12
    # upsert replaces
    db.index_solution(conn, "spotA", "solves/new.json", sg.SOLVER_VERSION, 0.003)
    row = conn.execute("SELECT * FROM solution_index WHERE spot_key='spotA'").fetchone()
    assert row["path"] == "solves/new.json"


# --------------------------------------------------------------------------- #
# Wave-2 [E26]: when a hero hand blocks the ENTIRE villain range, its valid
# opponent reach is 0, so every per-action EV divides to 0 and the average
# strategy falls back to uniform. The adapter emitted that as a real Solution:
# all EVs 0 and every frequency >= 5%, which the decision-e rule reads as
# "every action is correct, ev_loss 0". A fabricated answer key is worse than
# no answer key — the class must simply not be emitted.
# --------------------------------------------------------------------------- #
def _fully_blocked_solved():
    """Hero holds AhAs (blocking every villain combo) plus one live hand."""
    board = _cards("2c3d4h5s7c")
    r0 = _range(["AhAs", "KdKh"])
    r1 = _range(["AhKc", "AsQc"])   # every villain combo shares a card with AhAs
    cfg = sg.BetConfig(sizes=(0.75,), jam=False, max_raises=0)
    tree = sg.build_tree(board, pot0=10.0, stack=20.0, cfg=cfg)
    s = sg.SubgameSolver(tree, board, r0, r1, pot0=10.0)
    s.iterate(50)
    return s


def test_fully_blocked_hand_class_is_not_emitted_as_a_solution():
    s = _fully_blocked_solved()
    sols = root_solution_by_class(s, player=0, range_ctx="ctx")

    assert "AA" not in sols, "a fully-blocked class must not be emitted"
    assert "KK" in sols, "the unblocked class must still be solved"


def test_no_emitted_solution_is_all_zero_ev_with_uniform_frequencies():
    """The fabrication signature: zero EVs everywhere + a flat strategy."""
    s = _fully_blocked_solved()
    for cls, sol in root_solution_by_class(s, player=0, range_ctx="ctx").items():
        evs = [ev for ev, _ in sol.actions.values()]
        freqs = [f for _, f in sol.actions.values()]
        assert any(ev != 0.0 for ev in evs) or len(set(freqs)) > 1, (
            f"class {cls} looks fabricated: EVs {evs}, freqs {freqs}")
