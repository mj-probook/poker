"""Slice C — ICM-adjusted jam/fold directional property (impl doc §3 Slice C).

Same push/fold game, but payoffs are ICM-$ deltas instead of chips on a bubble
fixture (4 players, 3 paid). The risk premium of busting on the money bubble
must make BB's calling range strictly tighter than the chip-EV range at the same
effective depth. Directional only — ICM push/fold is general-sum, so we assert
the *direction*, never an exploitability number (grading-honesty rule).
"""

from pokerlab.charts import hands
from pokerlab.charts.jamfold import solve_jamfold, solve_jamfold_icm

# Bubble: 4 players, top 3 paid, bb = 100 chips.
_PAYOUTS = (500, 300, 200)
_BB_CHIPS = 100


def test_bubble_bb_calls_tighter_than_chip_ev_same_depth():
    stacks = (1000, 1000, 1000, 1000)  # all 10bb
    chip = solve_jamfold(10.0)
    icm = solve_jamfold_icm(stacks, sb_seat=0, bb_seat=1,
                            payouts=_PAYOUTS, bb_chips=_BB_CHIPS)
    assert icm.depth_bb == 10.0
    assert icm.bb_call_combos() < chip.bb_call_combos()


def test_shorter_covered_stack_tightens_bb_further():
    # A very short 4th stack raises bubble pressure -> BB calls even tighter.
    balanced = solve_jamfold_icm((1000, 1000, 1000, 1000), 0, 1, _PAYOUTS, _BB_CHIPS)
    pressured = solve_jamfold_icm((1500, 1000, 2000, 500), 0, 1, _PAYOUTS, _BB_CHIPS)
    assert pressured.bb_call_combos() < balanced.bb_call_combos()


def test_bb_still_calls_the_nuts_under_icm():
    # ICM tightens but never folds aces — a sanity floor on the risk premium.
    icm = solve_jamfold_icm((1000, 1000, 1000, 1000), 0, 1, _PAYOUTS, _BB_CHIPS)
    assert icm.bb_call[hands.HAND_INDEX["AA"]] >= 0.99


# --------------------------------------------------------------------------- #
# Round-1 finding [20]: the ICM solve reported exploitability=NaN — i.e. it was
# never verified at all, while feeding the LIVE M2 bubble drills. The solve is
# general-sum, so there is no zero-sum "nash_conv"; but the per-player
# best-response gap IS the Nash gap, and it is exactly what must be small for
# these ranges to be trustworthy answer keys.
# --------------------------------------------------------------------------- #
def test_icm_solve_reports_a_verified_nash_gap():
    icm = solve_jamfold_icm((1000, 1000, 1000, 1000), 0, 1, _PAYOUTS, _BB_CHIPS)
    assert icm.exploitability == icm.exploitability, "exploitability is NaN"
    assert icm.exploitability >= 0.0
    assert icm.exploitability < 1e-4


def test_icm_nash_gap_holds_on_the_pressured_bubble_fixture():
    icm = solve_jamfold_icm((1500, 1000, 2000, 500), 0, 1, _PAYOUTS, _BB_CHIPS)
    assert icm.exploitability < 1e-4


# --------------------------------------------------------------------------- #
# Round-2 findings [E34][E35][E36][E37]: the public chart entry points are
# answer keys for live drills and HH grading, so a degenerate input must fail
# loudly rather than converge to a confident-looking wrong chart.
# --------------------------------------------------------------------------- #
import math

import pytest

from pokerlab.charts.icm import icm_equities
from pokerlab.charts.jamfold import jamfold_range


def test_icm_nash_gap_is_invariant_under_payout_scaling():
    """[E37] the guard must measure convergence, not the ladder's units."""
    base = solve_jamfold_icm((1000, 1000, 1000, 1000), 0, 1, _PAYOUTS, _BB_CHIPS)
    scaled = solve_jamfold_icm((1000, 1000, 1000, 1000), 0, 1,
                               tuple(p * 100 for p in _PAYOUTS), _BB_CHIPS)
    assert scaled.exploitability == pytest.approx(base.exploitability, rel=1e-9)
    # ...and the identical solve on a realistic cents ladder still clears the bar
    cents = solve_jamfold_icm((1000, 1000, 1000, 1000), 0, 1,
                              (50000, 30000, 20000), _BB_CHIPS)
    assert cents.exploitability < 1e-4


@pytest.mark.parametrize("payouts", [(), (0, 0, 0), (200, 300, 500), (-100, 50)])
def test_degenerate_prize_ladders_are_rejected(payouts):
    """[E34] empty/zero ladders yield a uniform 50/50 chart at a genuine 0.0 gap."""
    with pytest.raises(ValueError):
        solve_jamfold_icm((1000, 1000, 1000, 1000), 0, 1, payouts, _BB_CHIPS)


@pytest.mark.parametrize("depth", [0.0, -10.0, float("nan"), float("inf")])
def test_non_positive_or_non_finite_depth_is_rejected(depth):
    """[E35] a negative depth converged to the MIRROR chart: fold AA, jam 32o."""
    with pytest.raises(ValueError):
        jamfold_range("SB", depth)


def test_same_seat_for_both_blinds_is_rejected():
    with pytest.raises(ValueError):
        solve_jamfold_icm((1000, 1000, 1000), 0, 0, _PAYOUTS, _BB_CHIPS)


def test_big_blind_shorter_than_one_blind_does_not_go_negative():
    """[E36] icm_model charged the nominal blind regardless of the seat's stack.

    A BB all-in for less than one big blind is routine on a bubble; subtracting
    a full blind drove the stack negative and icm_equities returned negative $.
    The forced all-in is the honest game, so the post is clamped to the stack.

    Completing at all is the assertion: without the clamp this solve feeds a
    negative stack to icm_equities, which now refuses it. Note the resulting
    chart calls ~100% -- that is CORRECT, not degenerate: at 0.4bb the BB is
    already all-in from posting the blind, so it has no fold to make. (An
    earlier version of this test asserted `bb_call_combos() < 1326`, which
    "passed" only because CFR+ converges to 1325.9994 -- a floating-point
    technicality that would not have caught a regression.)
    """
    sol = solve_jamfold_icm((1000, 40, 1000, 1000), 0, 1, _PAYOUTS, _BB_CHIPS)
    assert math.isfinite(sol.exploitability) and sol.exploitability >= 0.0
    assert sol.depth_bb == pytest.approx(0.4)


def test_icm_equities_rejects_a_negative_stack():
    with pytest.raises(AssertionError):
        icm_equities([100, -50, 100], [500, 300])


# --------------------------------------------------------------------------- #
# Round-2 finding [E50]: the ICM chart's near-universal jamming was confirmed
# GENUINE, not a solver artefact -- but that verdict rested on the numpy solve
# and its own exploitability instrument, which share an implementation and so
# cannot corroborate each other. This judges the numpy profile with the
# independent Slice-B game-tree walk (cfr.exploit.nash_conv) at full 169-class
# scale on the ICM model.
# --------------------------------------------------------------------------- #
def test_numpy_exploitability_matches_an_independent_tree_walk():
    """Two independent implementations must agree on the ICM Nash gap.

    `model_exploitability` is closed-form numpy over the 169x169 payoff
    matrices; `nash_conv` walks a real 28,561-chance-outcome JamFoldGame tree
    built from the same model. They share no code below JamFoldModel, so
    agreement pins BOTH the numpy solve and the exploitability math -- either
    one drifting breaks this. The assertion is on their RATIO, so it is
    indifferent to legitimate movement in the ranges themselves.

    nash_conv is the summed per-player best-response gain; exploitability is
    that divided by num_players (the pyspiel convention, impl doc §1), hence
    the /2 for this 2-player game.

    Subsumes test_charts_jamfold.py::test_numpy_solver_matches_cfr_on_tiny_fixture,
    which does the same cross-check on 3 hand classes of the CHIP model -- this
    is the full 169 classes on the ICM model, where the general-sum payoffs make
    the two implementations much easier to disagree.

    Deliberately NOT asserted here: a threshold on sb_jam. The solved profile
    jams ~everything, but CFR+ converges to 0.999996, not 1.0 -- exact 1.0 is
    unreachable in finite iterations, so `assert sb_jam.min() < 1.0` would pass
    on floating-point convergence noise while proving nothing about the chart.
    Agreement between two implementations is the load-bearing claim.
    """
    from pokerlab.cfr import build_tree, nash_conv
    from pokerlab.charts.equity import load_equity_matrix
    from pokerlab.charts.jamfold import (
        JamFoldGame,
        icm_model,
        joint_prior,
        model_exploitability,
        solve_model,
    )

    E = load_equity_matrix().equity_matrix
    model = icm_model([10000] * 4, 0, 1, list(_PAYOUTS), 1000, 0.0, E)
    P = joint_prior()
    P = P / P.sum()

    x, y, w = solve_model(model, P, iters=1500)
    expl_np = model_exploitability(model, P, x, y, w)
    assert expl_np > 0.0, "a zero gap would make the ratio assertion vacuous"

    tree = build_tree(JamFoldGame(model, P))
    profile = {}
    for i in range(len(x)):
        profile[f"SB:{i}"] = {"jam": float(x[i]), "fold": float(1.0 - x[i])}
        profile[f"BB:{i}"] = {"call": float(y[i]), "fold": float(1.0 - y[i])}

    independent = nash_conv(tree, profile) / 2.0      # -> per-player gap
    assert abs(independent - expl_np) / expl_np < 1e-6


# --------------------------------------------------------------------------- #
# Round-3 findings [C1][C2][C3][C4]: the chart entry points are answer keys, so
# every degenerate input must fail loudly. Wave 2 closed four such holes; these
# are the ones that survived it.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("stacks,seat", [((1000, 0, 1000, 1000), "bb_seat"),
                                         ((0, 1000, 1000, 1000), "sb_seat")])
def test_icm_rejects_a_blind_seat_with_no_chips(stacks, seat):
    """[C1] the chip path refused depth 0; the ICM path served a uniform chart.

    With a 0-chip blind the effective depth is 0, every hand is equivalent, and
    the solve certifies itself at ~5e-9 -- AA and 32o came back identical. The
    asymmetry was the tell: solve_jamfold(0.0) already raised.
    """
    with pytest.raises(ValueError, match="no chips"):
        solve_jamfold_icm(stacks, 0, 1, _PAYOUTS, _BB_CHIPS)
    with pytest.raises(ValueError):          # the chip path, for contrast
        solve_jamfold(0.0)


@pytest.mark.parametrize("stacks", [[float("inf"), 100, 100],
                                    [float("nan"), 100, 100]])
def test_icm_equities_rejects_non_finite_stacks(stacks):
    """[C2] `s >= 0` did not express the intent: `inf >= 0` is True, and an
    infinite stack turns stack/total into a NaN that spreads silently."""
    with pytest.raises(AssertionError, match="finite"):
        icm_equities(stacks, [500, 300])


@pytest.mark.parametrize("payouts", [[500, -300], [float("inf"), 300],
                                     [500, float("nan")]])
def test_icm_equities_rejects_bad_ladders(payouts):
    with pytest.raises(AssertionError, match="payouts"):
        icm_equities([100, 100, 100], payouts)


@pytest.mark.parametrize("ante", [float("nan"), float("inf"), -0.5])
def test_non_finite_or_negative_ante_is_rejected(ante):
    """[C4] the ante enters the payoffs like the depth does: NaN poisons every
    payoff, and a negative ante removes chips from the pot."""
    with pytest.raises(ValueError, match="ante"):
        jamfold_range("SB", 10.0, ante)
    with pytest.raises(ValueError, match="ante"):
        solve_jamfold_icm((1000,) * 4, 0, 1, _PAYOUTS, _BB_CHIPS, ante)


def test_exploitability_normalizes_by_reachable_prizes_only():
    """[C3] ICM pays min(#payouts, #players) places; counting unreachable
    prizes inflates the denominator and deflates the guard (1.15x measured)."""
    stacks = (1000, 1000, 1000)                     # 3 players...
    ladder = (500, 300, 200, 100, 50)               # ...5-place ladder
    sol = solve_jamfold_icm(stacks, 0, 1, ladder, _BB_CHIPS)
    reachable = float(sum(ladder[:len(stacks)]))
    raw = sol.exploitability * reachable
    # normalizing by the FULL ladder would report a 1.15x smaller gap
    assert sol.exploitability == pytest.approx(raw / reachable, rel=1e-12)
    assert sol.exploitability > raw / float(sum(ladder))
