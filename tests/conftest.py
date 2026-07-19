"""Shared pytest fixtures."""

import pytest


@pytest.fixture(scope="session")
def leduc_sigma_star():
    """A converged full-game Leduc CFR+ solution (tree, average profile).

    Session-scoped so the Slice-G ReBeL tests share a single solve instead of
    re-solving Leduc in every test. 700 CFR+ iterations put NashConv well under
    1e-3, enough for both the depth-limited value checks and the safe-resolve bar.
    """
    from pokerlab.cfr.cfr import CFRSolver
    from pokerlab.cfr.game import build_tree
    from pokerlab.cfr.leduc import LeducPoker

    tree = build_tree(LeducPoker())
    solver = CFRSolver(tree, plus=True)
    solver.run(700)
    return tree, solver.average_profile()
