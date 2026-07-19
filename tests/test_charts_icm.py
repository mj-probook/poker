"""Slice C — Malmuth-Harville ICM (impl doc §3 Slice C; plan §4 ICM).

Every expected value below is derived by hand in the comment above its
assertion (no vendor data). Tolerance 0.1% per the M1.5 exit criterion.
"""

import math

import pytest

from pokerlab.charts import icm_equities


def _rel(actual, expected, rel=1e-3):
    assert actual == pytest.approx(expected, rel=rel), (actual, expected)


def test_three_player_two_payouts_hand_computed():
    # stacks [50,30,20] (S=100), payouts [70,30]. P(i first)=s_i/100.
    # P0 = 0.5*70 + [0.3*(50/70)+0.2*(50/80)]*30 = 35 + 10.178571 = 45.178571
    # P1 = 0.3*70 + [0.5*(30/50)+0.2*(30/80)]*30 = 21 + 11.25     = 32.25
    # P2 = 0.2*70 + [0.5*(20/50)+0.3*(20/70)]*30 = 14 + 8.571429  = 22.571429
    eq = icm_equities([50, 30, 20], [70, 30])
    _rel(eq[0], 45.178571)
    _rel(eq[1], 32.25)
    _rel(eq[2], 22.571429)
    _rel(sum(eq), 100.0)  # equities sum to total prize pool


def test_three_player_three_payouts_hand_computed():
    # stacks [50,30,20], payouts [50,30,20] — full ladder, 3rd = last place.
    # P0 = 25 + 10.178571 + (1-0.5-0.339286)*20 = 25+10.178571+3.214286 = 38.392857
    # P1 = 15 + 11.25     + (1-0.3-0.375)*20     = 32.75
    # P2 = 10 + 8.571429  + (1-0.2-0.285714)*20  = 28.857143
    eq = icm_equities([50, 30, 20], [50, 30, 20])
    _rel(eq[0], 38.392857)
    _rel(eq[1], 32.75)
    _rel(eq[2], 28.857143)


def test_equal_stacks_are_symmetric():
    # Identical stacks -> identical equity = (sum of payouts)/N regardless of ladder.
    eq = icm_equities([100, 100, 100], [60, 30, 10])
    for x in eq:
        _rel(x, 100.0 / 3.0)


def test_bubble_equal_stacks_four_players_three_paid():
    # 4 equal stacks, 3 paid: each equity = 100/4 = 25 (symmetry).
    eq = icm_equities([1000, 1000, 1000, 1000], [50, 30, 20])
    for x in eq:
        _rel(x, 25.0)
    _rel(sum(eq), 100.0)


def test_single_player_takes_first_place_only():
    # One player left, two payouts posted: gets 1st only (no 2nd place exists).
    eq = icm_equities([500], [100, 50])
    assert eq == pytest.approx([100.0])


def test_more_payouts_than_players_truncates():
    # 2 players, 3 payouts -> only top 2 are reachable.
    # P0 = 0.6*50 + 0.4*30 = 42 ; P1 = 0.4*50 + 0.6*30 = 38.
    eq = icm_equities([60, 40], [50, 30, 20])
    _rel(eq[0], 42.0)
    _rel(eq[1], 38.0)
    _rel(sum(eq), 80.0)  # only 50+30 distributable with 2 players


def test_chip_leader_has_more_equity_but_less_than_chip_share():
    # ICM compresses: leader equity share < chip share (the core ICM fact).
    stacks = [70, 20, 10]
    eq = icm_equities(stacks, [65, 25, 10])
    chip_share = 70 / 100
    equity_share = eq[0] / sum(eq)
    assert equity_share < chip_share
    assert eq[0] > eq[1] > eq[2]  # monotone in stack


def test_equities_sum_to_prize_pool_property():
    # Random-ish stacks: equities always sum to the distributable prize pool.
    stacks = [37, 29, 18, 11, 5]
    payouts = [50, 30, 20]
    eq = icm_equities(stacks, payouts)
    assert math.isclose(sum(eq), sum(payouts), rel_tol=1e-9)
