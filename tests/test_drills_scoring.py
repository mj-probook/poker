"""Slice E — decision-ε scoring rule (impl doc §1; plan §1 definitions).

correct iff  ev_loss <= max(0.005*pot_bb, 0.1)  OR  chosen frequency >= 0.05.
Every case below constructs a Solution by hand so the rule is exercised in
isolation from the chart engine.
"""

import pytest

from pokerlab.drills.scoring import score
from pokerlab.types import Score, Solution


def _sol(actions):
    return Solution(actions=actions, range_ctx="test", source="chart")


def test_clear_best_action_scores_correct_zero_loss():
    sol = _sol({"jam": (0.42, 0.97), "fold": (0.0, 0.03)})
    s = score(sol, "jam", pot_bb=20.0)
    assert isinstance(s, Score)
    assert s.correct
    assert s.ev_loss_bb == pytest.approx(0.0)
    assert s.best_action == "jam"
    assert s.chosen_frequency == pytest.approx(0.97)


def test_small_loss_under_floor_is_correct():
    # chosen loses 0.08bb (< 0.1bb floor) and has ~0 frequency -> correct by floor.
    sol = _sol({"call": (1.00, 0.0), "fold": (0.92, 1.0)})
    s = score(sol, "fold", pot_bb=2.0)   # eps = max(0.01, 0.1) = 0.1
    assert s.correct
    assert s.ev_loss_bb == pytest.approx(0.08)
    assert s.best_action == "call"


def test_loss_over_floor_but_frequency_accepts_mixed_spot():
    # chosen loses 0.30bb (> 0.1 floor, > 0.5% pot) but is a 6% mix -> correct.
    sol = _sol({"jam": (1.00, 0.94), "fold": (0.70, 0.06)})
    s = score(sol, "fold", pot_bb=20.0)  # eps = max(0.1, 0.1) = 0.1
    assert s.ev_loss_bb == pytest.approx(0.30)
    assert s.chosen_frequency == pytest.approx(0.06)
    assert s.correct           # frequency >= 0.05 rescues it


def test_both_fail_is_incorrect():
    # loss 0.30bb over floor AND frequency 0.02 < 0.05 -> incorrect.
    sol = _sol({"jam": (1.00, 0.98), "fold": (0.70, 0.02)})
    s = score(sol, "fold", pot_bb=20.0)
    assert not s.correct
    assert s.ev_loss_bb == pytest.approx(0.30)
    assert s.best_action == "jam"
    assert s.chosen_frequency == pytest.approx(0.02)


def test_pot_scaled_epsilon_dominates_floor_in_big_pots():
    # In a 100bb pot eps = 0.5bb > 0.1 floor: a 0.4bb loss now clears.
    sol = _sol({"bet": (2.0, 0.9), "check": (1.6, 0.0)})
    s = score(sol, "check", pot_bb=100.0)
    assert s.ev_loss_bb == pytest.approx(0.4)
    assert s.correct           # 0.4 <= 0.005*100 = 0.5


def test_missing_action_label_raises():
    sol = _sol({"jam": (0.42, 0.97), "fold": (0.0, 0.03)})
    with pytest.raises(ValueError):
        score(sol, "raise", pot_bb=20.0)


# --------------------------------------------------------------------------- #
# Round-3 finding [P7']: ICM drills carry $-denominated EVs and were graded
# against a bb-denominated ε. Both acceptance branches were dead — the ε branch
# by unit mismatch (a 109$ delta vs a 0.1bb floor), the 5% mixed-spot hatch
# because the ICM solve is near-pure — leaving exact-argmax grading with no
# indifference tolerance at all. `bb_value` converts the SAME rule into the
# decision's own currency rather than inventing a second tolerance constant.
# --------------------------------------------------------------------------- #
def test_bb_value_converts_epsilon_into_the_payoff_currency():
    from pokerlab.drills.scoring import epsilon

    # 20bb pot: eps = max(0.005*20, 0.1) = 0.1bb. At 25$/bb that is 2.50$.
    assert epsilon(20.0) == pytest.approx(0.1)
    assert epsilon(20.0, bb_value=25.0) == pytest.approx(2.5)
    # scale-free: rescaling the payoff unit moves eps with it, so the same
    # decision grades identically in cents or dollars ([E37] normalize-by-pool).
    assert epsilon(20.0, bb_value=2500.0) == pytest.approx(250.0)


def test_a_near_indifferent_icm_spot_is_not_graded_wrong():
    """The behaviour the unit mismatch destroyed: $-indifference is tolerated."""
    # A 2$ mistake at 25$/bb is 0.08bb — inside the 0.1bb floor, so correct.
    sol = _sol({"jam": (50.0, 1.0), "fold": (48.0, 0.0)})
    assert score(sol, "fold", pot_bb=20.0, bb_value=25.0).correct
    # ...while a genuinely large $ error is still wrong at the same conversion.
    big = _sol({"jam": (50.0, 1.0), "fold": (10.0, 0.0)})
    assert not score(big, "fold", pot_bb=20.0, bb_value=25.0).correct


def test_without_the_conversion_real_icm_grading_is_exact_argmax():
    """Guards the tests above from going vacuous — on the REAL solve, not a toy.

    A synthetic near-zero delta proves nothing here: 0.001 clears the 0.1bb
    floor whatever the units, so it would pass unconverted and make this look
    like a no-op. The claim is about ACTUAL ICM magnitudes, where $-deltas run
    to ~110 against a 0.1bb floor, so it has to be measured on the real drills.
    """
    from pokerlab.drills import generator as gen

    icm = gen.icm_drills()
    non_best = [(d, a) for d in icm for a in d.legal_actions
                if score(d.solution, a, d.pot_bb, bb_value=d.bb_value).ev_loss_bb > 0]
    assert non_best, "fixture has no non-best actions — test is vacuous"
    unconverted = sum(1 for d, a in non_best if score(d.solution, a, d.pot_bb).correct)
    converted = sum(1 for d, a in non_best
                    if score(d.solution, a, d.pot_bb, bb_value=d.bb_value).correct)
    assert unconverted == 0, "expected exact-argmax grading without the conversion"
    assert converted > 0, "conversion restored no tolerance on the real solve"


def test_icm_drills_carry_the_conversion_and_chip_drills_do_not():
    """The factor must reach the scorer from the drill, not the call site."""
    from pokerlab.drills import generator as gen

    icm = gen.icm_drills()
    tc = gen.BUBBLE
    expected = sum(tc.payouts) / (sum(tc.stacks_all) / tc.bb)
    assert all(d.bb_value == pytest.approx(expected) for d in icm)
    assert expected == pytest.approx(25.0)          # pool 1000 over 40bb
    assert all(d.bb_value == 1.0 for d in gen.jamfold_drills()[:50])
    # and the conversion actually restores tolerance on real ICM solutions
    tolerated = sum(
        1 for d in icm for a in d.legal_actions
        if (s := score(d.solution, a, d.pot_bb, bb_value=d.bb_value)).ev_loss_bb > 0
        and s.correct)
    assert tolerated > 0, "conversion did not restore any indifference tolerance"
