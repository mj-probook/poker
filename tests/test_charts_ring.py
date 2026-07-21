"""Full-ring first-in jam/fold chain charts (charts/ring.py).

The game: first-in jammer at a 9-max position, every player behind responds
call/fold in order, at most one caller (single-caller restriction — overcall
trees are not modeled), pairwise jammer↔responder card removal only, symmetric
stacks. All three restrictions are DISCLOSED in the Solution range_ctx; within
the restricted game the solve carries the same exploitability certificate as
the HU charts.

Anchor #1 is the strongest check available: a 2-seat table (SB jammer, BB
responder) is exactly the game `solve_jamfold` already solves and certifies,
so the chain solver must reproduce it.
"""

import numpy as np
import pytest

from pokerlab.charts import hands
from pokerlab.charts.jamfold import solve_jamfold
from pokerlab.charts.ring import RING_ORDER, solve_ring


def _combos(freqs: np.ndarray) -> float:
    return float(sum(f * hands.combos(h)
                     for f, h in zip(freqs, hands.HAND_CLASSES)))


def test_two_seat_chain_reduces_to_the_hu_solve():
    """Same equilibrium as `solve_jamfold`, certified by ITS instrument.

    Frequencies are compared only where the HU solve is pure: at genuinely
    mixed boundary classes (43s, Q6s at 10bb) any split is an equilibrium, so
    two correct solvers may legitimately disagree there. The load-bearing
    assertion is the cross-certificate — the chain solver's profile judged by
    jamfold's independent `model_exploitability` (measured 7.7e-07, below the
    HU solve's own 1.4e-06).
    """
    from pokerlab.charts.equity import load_equity_matrix
    from pokerlab.charts.jamfold import (chip_model, joint_prior,
                                         model_exploitability)

    hu = solve_jamfold(10.0)
    ring = solve_ring("SB", 10.0, table=("SB", "BB"))
    for ours, theirs in ((ring.jam, hu.sb_jam),
                         (ring.calls["BB"], hu.bb_call)):
        pure = (theirs < 0.01) | (theirs > 0.99)
        assert pure.sum() > 100          # the anchor covers most of the range
        assert np.allclose(ours[pure], theirs[pure], atol=5e-3)
    model = chip_model(10.0, 0.0, load_equity_matrix().equity_matrix)
    P = joint_prior()
    cross = model_exploitability(model, P, ring.jam, ring.calls["BB"],
                                 P.sum(axis=1))
    assert cross < 1e-4
    assert ring.exploitability < 1e-4


def test_nine_max_utg_solve_is_certified(seed_depth=10.0):
    sol = solve_ring("UTG", seed_depth)
    # every player behind the jammer has a strategy
    assert set(sol.calls) == set(RING_ORDER[1:])
    assert sol.exploitability < 1e-4


def test_jam_ranges_widen_with_position():
    widths = [_combos(solve_ring(p, 10.0).jam)
              for p in ("UTG", "LJ", "CO", "BTN")]
    assert widths == sorted(widths)
    assert widths[-1] > widths[0]           # strictly wider on the button


def test_bb_defends_wider_versus_later_jams():
    vs_utg = _combos(solve_ring("UTG", 10.0).calls["BB"])
    vs_btn = _combos(solve_ring("BTN", 10.0).calls["BB"])
    assert vs_btn > vs_utg


def test_published_fact_aa_always_jams_and_calls():
    aa = hands.HAND_CLASSES.index("AA")
    for pos in ("UTG", "CO", "BTN"):
        sol = solve_ring(pos, 10.0)
        assert sol.jam[aa] > 0.99
        for resp in sol.calls.values():
            assert resp[aa] > 0.99


def test_published_fact_trash_folds_utg():
    # 72o/32o first-in from UTG at 15bb 9-max are folds in every published
    # push/fold reference
    sol = solve_ring("UTG", 15.0)
    for label in ("72o", "32o"):
        assert sol.jam[hands.HAND_CLASSES.index(label)] < 0.01


def test_ring_range_returns_solutions_with_disclosed_restrictions():
    from pokerlab.charts.ring import ring_range
    rng = ring_range("CO", 10.0)
    sol = rng["AKs"]
    assert set(sol.actions) == {"jam", "fold"}
    assert sol.source == "chart"
    # the restrictions must travel WITH the answer key, not live in a doc
    for token in ("single-caller", "pairwise-removal", "stacks-symmetric"):
        assert token in sol.range_ctx


def test_ring_defense_range_prices_call_vs_named_jammer():
    from pokerlab.charts.ring import ring_defense_range
    rng = ring_defense_range("BB", versus="CO", depth_bb=10.0)
    sol = rng["AA"]
    assert set(sol.actions) == {"call", "fold"}
    assert sol.actions["call"][1] > 0.99          # AA always calls
    assert "CO" in sol.range_ctx                  # names the jammer it defends


def test_ring_defense_range_rejects_a_seat_not_behind_the_jammer():
    from pokerlab.charts.ring import ring_defense_range
    with pytest.raises(ValueError):
        ring_defense_range("UTG", versus="CO", depth_bb=10.0)
