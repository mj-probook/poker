"""The DEFAULT bet grid must build trees at the flops25 geometry.

Latent [M11]-adjacent bug found while wiring postflop drills (2026-07-29):
a 0.33-pot bet at pot0=5.0 leaves both stacks at 33.111 via two different
float paths (bettor: stack−bet; caller: stack−to_call), and the symmetric-
stack assert compared with `==`, so the whole default grid crashed on tree
build. Single-size test grids (0.75 of small integer pots) happened to stay
exactly equal, which is why every existing test passed.
"""

import pytest

from pokerlab.engine.cards import card_from_str
from pokerlab.solver import subgame as sg


def _cards(s: str) -> tuple[int, ...]:
    return tuple(card_from_str(s[i:i + 2]) for i in range(0, len(s), 2))


def test_033_size_builds_through_a_chance_node_at_flops25_geometry():
    """The minimal reproduction: a 0.33-pot bet at pot0=5.0/stack=37.5 makes
    the two float paths to the next chance node disagree in the last bits.
    A TURN tree hits the identical assert in milliseconds; the full default-
    grid FLOP tree (51s to build) runs behind `slow` below."""
    tree = sg.build_tree(_cards("2c3c4c8d"), pot0=5.0, stack=37.5,
                         cfg=sg.BetConfig(sizes=(0.33,), jam=True,
                                          max_raises=2))
    assert tree is not None


@pytest.mark.slow
def test_default_grid_builds_at_flops25_geometry():
    tree = sg.build_tree(_cards("2c3c4c"), pot0=5.0, stack=37.5,
                         cfg=sg.BetConfig())
    assert tree is not None


@pytest.mark.slow  # ~15s: 1 CFR iter + exploitability on the full-grid tree
def test_default_grid_values_flow_end_to_end_at_turn_scale():
    """Values flow through DEFAULT-GRID multi-size bet nodes + a chance node,
    with the fixture ranges. Deliberately TURN scale: the same smoke at flop
    scale was measured at >10 minutes for one iteration + one measurement
    (thousands of runout showdown contexts) — that run belongs to the
    nightly batch drain, not to any test tier."""
    from pokerlab.solver import flops25
    oop, ip = flops25.range_vectors()
    board = _cards("2c3c4c8d")
    tree = sg.build_tree(board, pot0=5.0, stack=37.5, cfg=sg.BetConfig())
    solver = sg.SubgameSolver(tree, board, oop, ip, pot0=5.0)
    solver.iterate(1)
    assert solver.exploitability() >= 0.0
