"""The checked-in ring chart artifact (data/ring_charts.npz).

Solving all 105 formations takes ~76s, so the charts ship as a self-generated
artifact (the equity-matrix pattern: built by scripts/gen_ring_charts.py,
checked in, regenerable). The honesty obligation is that the artifact never
becomes a black box: the fast suite RE-CERTIFIES every stored formation from
scratch — the Nash gap is recomputed from the cached strategies against a
freshly built payoff model, no solver state involved — so a corrupted or
stale artifact fails loudly. A sampled live re-solve runs behind `slow`.
"""

import numpy as np
import pytest

from pokerlab.charts.ring import (RING_JAMMERS, RING_ORDER, _Chain,
                                  _nash_gap, load_ring_charts, ring_solution,
                                  solve_ring)
from pokerlab.drills.categories import ANTES, DEPTHS


def test_artifact_covers_the_full_drill_grid():
    charts = load_ring_charts()
    assert len(charts) == len(RING_JAMMERS) * len(DEPTHS) * len(ANTES)
    for jammer in RING_JAMMERS:
        for depth in DEPTHS:
            for ante in ANTES:
                sol = ring_solution(jammer, float(depth), float(ante))
                assert sol.jammer == jammer
                behind = RING_ORDER[RING_ORDER.index(jammer) + 1:]
                assert set(sol.calls) == set(behind)


def test_every_cached_formation_recertifies_from_scratch():
    """The certificate must hold for what is actually SHIPPED, recomputed
    against a fresh payoff model — not trusted from generation time."""
    from pokerlab.charts.equity import load_equity_matrix
    from pokerlab.charts.jamfold import joint_prior

    E = load_equity_matrix().equity_matrix
    P = joint_prior()
    worst = 0.0
    for sol in load_ring_charts().values():
        chain = _Chain(sol.table, sol.jammer, sol.depth_bb, sol.ante, E, P)
        gap = _nash_gap(chain, sol.jam, sol.calls)
        worst = max(worst, gap)
    # The solver's stopping rule is target_gap=1e-5 and the shipped grid's
    # worst measured gap is 9.97e-06. The 1.5x is headroom for float drift
    # across platforms, not for behavior ([R4-3]: a bar orders of magnitude
    # above the measured value stays green through real regressions).
    assert worst < 1.5e-5


@pytest.mark.slow
def test_cached_matches_a_live_resolve():
    cached = ring_solution("CO", 10.0, 0.125)
    live = solve_ring("CO", 10.0, 0.125)
    assert np.allclose(cached.jam, live.jam, atol=1e-8)
    for q in cached.calls:
        assert np.allclose(cached.calls[q], live.calls[q], atol=1e-8)
