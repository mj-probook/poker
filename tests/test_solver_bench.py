"""Slice D — milestone-exit benchmark (impl doc §3 Slice D; plan M3).

Marked ``bench`` (excluded from the fast suite; run via ``make bench``).

Two things are demonstrated on the checked-in flops25 fixture with its real
(wide) BTNopen_BBcall ranges:

  * **Accuracy bar** (≤0.5% pot exploitability, own verified BR instrument) on
    fixture-flop river subgames — tractable in pure Python and cleared handily.
  * **Full flop subgame** solving runs correctly (two chance layers) and drives
    exploitability down, but reaching the 0.5% bar over 49×48 runouts is
    ~1 hour/flop in pure Python — this is precisely the hot loop the plan
    schedules for a Rust port at M3. The test records the trajectory rather than
    waiting out the full convergence.
"""

from __future__ import annotations

import pytest

from pokerlab.engine.cards import card_from_str
from pokerlab.solver import subgame as sg
from pokerlab.solver.flops25 import build_solver, load_flops25, range_vectors
from pokerlab.solver.solve_io import insert_solution_index, write_solve
from pokerlab.store import db

pytestmark = pytest.mark.bench


def _complete_to_river(flop: tuple[int, ...]) -> tuple[int, ...]:
    extra = [c for c in range(52) if c not in flop][:2]
    return flop + tuple(extra)


@pytest.mark.parametrize("idx", [0, 1, 6, 12, 18])  # spread across texture classes
def test_fixture_river_spots_meet_accuracy_bar(idx, tmp_path):
    fx = load_flops25()
    oop, ip = range_vectors(fx)
    entry = fx["flops"][idx]
    flop = tuple(card_from_str(entry["cards"][i:i + 2]) for i in range(0, 6, 2))
    board = _complete_to_river(flop)
    cfg = sg.BetConfig(sizes=(0.75,), jam=False, max_raises=1)
    tree = sg.build_tree(board, pot0=fx["meta"]["pot_bb"],
                         stack=fx["meta"]["eff_stack_bb"], cfg=cfg)
    s = sg.SubgameSolver(tree, board, oop.copy(), ip.copy(), pot0=fx["meta"]["pot_bb"])
    s.iterate(600)
    expl = s.exploitability_pct()
    assert expl < 0.005, f"flop {entry['cards']} river expl {expl:.4%} > 0.5%"

    # exercise the persistence path at fixture scale for one spot
    spot_key = f"{entry['iso_class']}:{entry['texture']}|40|BTNopen_BBcall|river"
    path, expl_bb = write_solve(s, spot_key=spot_key, path=tmp_path / "solve.json")
    conn = db.connect(":memory:")
    insert_solution_index(conn, spot_key, path, sg.SOLVER_VERSION, expl_bb / s.pot0)
    assert conn.execute("SELECT count(*) c FROM solution_index").fetchone()["c"] == 1


def test_flop_subgame_solves_and_drives_exploitability_down():
    # Full flop subgame (turn+river chance). Correctness + downward trend on a
    # capped budget; the 0.5% bar over all runouts is the Rust-port target (M3).
    fx = load_flops25()
    cfg = sg.BetConfig(sizes=(0.75,), jam=False, max_raises=0)
    s = build_solver(fx["flops"][0], cfg=cfg)
    s.iterate(2)
    early = s.exploitability_pct()
    s.iterate(13)  # ~15 total; each iter walks 49×48 runouts
    late = s.exploitability_pct()
    print(f"flop {fx['flops'][0]['cards']}: expl {early:.4f} -> {late:.4f} over 15 iters")
    assert late < early  # CFR+ is reducing exploitability on the full flop tree
