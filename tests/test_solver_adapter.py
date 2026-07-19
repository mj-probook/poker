"""Slice D — Solution adapter + solve writer + solution_index (impl §3 Slice D)."""

from __future__ import annotations

import numpy as np

from pokerlab.engine.cards import card_from_str
from pokerlab.solver import subgame as sg
from pokerlab.solver.adapter import root_solution_by_class
from pokerlab.solver.solve_io import insert_solution_index, load_solve, write_solve
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
    insert_solution_index(conn, "spotA", "solves/spotA.json", sg.SOLVER_VERSION, 0.004)
    row = conn.execute("SELECT * FROM solution_index WHERE spot_key='spotA'").fetchone()
    assert row["path"] == "solves/spotA.json"
    assert row["solver_version"] == sg.SOLVER_VERSION
    assert abs(row["exploitability"] - 0.004) < 1e-12
    # upsert replaces
    insert_solution_index(conn, "spotA", "solves/new.json", sg.SOLVER_VERSION, 0.003)
    row = conn.execute("SELECT * FROM solution_index WHERE spot_key='spotA'").fetchone()
    assert row["path"] == "solves/new.json"
