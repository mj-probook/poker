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
