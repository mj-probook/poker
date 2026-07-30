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


@pytest.mark.slow  # a single full-grid FLOP iteration is ~30s — not fast-suite
def test_flops25_fixture_solver_builds_and_solves_a_little():
    from pokerlab.solver import flops25
    entry = flops25.load_flops25()["flops"][0]
    solver = flops25.build_solver(entry)
    solver.iterate(1)                     # smoke: values flow end to end
    assert solver.exploitability() >= 0.0
