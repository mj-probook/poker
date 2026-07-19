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
