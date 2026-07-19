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
