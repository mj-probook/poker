"""Slice A: NLHE state machine — posting, legality, side pots, showdown, replay.

Outcomes here are hand-computed; the PokerKit differential (test_engine_diff)
is the exhaustive oracle.
"""

import pytest

from pokerlab.engine.cards import card_from_str as C
from pokerlab.engine.state import new_hand


def _hole(*pairs):
    return [(C(a), C(b)) for a, b in pairs]


def _board(*cs):
    return [C(c) for c in cs]


def test_posts_antes_and_blinds() -> None:
    h = new_hand([2000, 2000, 2000], button=2, bb=100, sb=50, ante=25,
                 hole=[None, None, None])
    assert h.contrib == [75, 125, 25]  # SB+ante, BB+ante, ante
    assert h.stack_left == [1925, 1875, 1975]
    assert h.street_bet == [50, 100, 0]
    assert h.current_bet == 100 and h.min_raise_inc == 100
    assert h.to_act == 2  # (button+3)%3 = UTG = button in 3-handed
    assert h.pot == 225


def test_legal_actions_menu() -> None:
    h = new_hand([1000, 1000], button=1, bb=100, sb=50, hole=[None, None])
    # HU: button/SB (seat1) acts first, facing the BB.
    acts = dict(h.legal_actions())
    assert acts["fold"] == 0
    assert acts["call"] == 100  # complete to the big blind
    assert acts["raise"] == 200  # min raise-to
    assert acts["allin"] == 1000


def test_hu_showdown_best_hand_wins() -> None:
    h = new_hand([1000, 1000], button=1, bb=100, sb=50,
                 hole=_hole(("As", "Ac"), ("Kd", "Kc")),
                 board=_board("2c", "7d", "9h", "Js", "Qs"))
    h.apply(("call", 100))   # seat1 SB completes
    h.apply(("check", 0))    # seat0 BB checks
    for _ in range(6):       # check down flop, turn, river
        h.apply(("check", 0))
    assert h.is_terminal()
    assert h.final_stacks() == [1100, 900]
    assert h.payoffs() == [100, -100]
    assert sum(h.payoffs()) == 0


def test_fold_returns_uncalled_bet() -> None:
    h = new_hand([1000, 1000], button=1, bb=100, sb=50, hole=[None, None])
    h.apply(("raise", 300))  # seat1 raises to 300
    h.apply(("fold", 0))     # seat0 folds
    assert h.is_terminal()
    assert h.payoffs() == [-100, 100]
    assert h.final_stacks() == [900, 1100]


def test_min_raise_reopen_and_illegal_shortraise() -> None:
    h = new_hand([1000, 1000], button=1, bb=100, sb=50, hole=[None, None])
    h.apply(("raise", 300))          # inc 200 -> min_raise 200
    assert h.raise_bounds() == (500, 1000)  # next raise-to >= 500
    with pytest.raises(ValueError):
        h.apply(("raise", 400))      # short of a full raise
    h.apply(("raise", 500))          # legal
    assert h.current_bet == 500


def test_three_way_side_pot() -> None:
    h = new_hand([100, 200, 300], button=2, bb=100, sb=50,
                 hole=_hole(("Ac", "Ad"), ("Kc", "Kd"), ("Qc", "Qd")),
                 board=_board("2h", "7d", "8s", "3c", "4h"))
    h.apply(("raise", 300))   # seat2 button jams to 300
    h.apply(("allin", 100))   # seat0 SB all-in for less
    h.apply(("allin", 200))   # seat1 BB all-in for less
    assert h.is_terminal()
    # AA wins main (300), KK wins side (200), seat2's uncalled 100 returned.
    assert h.final_stacks() == [300, 200, 100]
    assert h.payoffs() == [200, 0, -200]


def test_split_pot_even() -> None:
    h = new_hand([1000, 1000], button=1, bb=100, sb=50,
                 hole=_hole(("Ac", "Kc"), ("Ad", "Kd")),
                 board=_board("2h", "5s", "9d", "Jh", "Qs"))
    h.apply(("call", 100)); h.apply(("check", 0))
    for _ in range(6):
        h.apply(("check", 0))
    assert h.final_stacks() == [1000, 1000]
    assert h.payoffs() == [0, 0]


def test_odd_chip_to_earliest_from_sb() -> None:
    # bb=75 -> a limped pot is 225; seat0 & seat1 tie -> 112/113 split.
    h = new_hand([1000, 1000, 1000], button=2, bb=75, sb=37,
                 hole=_hole(("Ac", "Kc"), ("Ad", "Kd"), ("2h", "3h")),
                 board=_board("As", "Ks", "9d", "4c", "7s"))
    h.apply(("call", 75))    # seat2 button
    h.apply(("call", 75))    # seat0 SB completes
    h.apply(("check", 0))    # seat1 BB option
    for _ in range(9):       # check down (3 players x 3 streets)
        if not h.is_terminal():
            h.apply(("check", 0))
    assert h.is_terminal()
    # 225 / 2 = 112 r1; odd chip to seat0 (SB-most winner).
    assert h.final_stacks() == [1038, 1037, 925]
    assert sum(h.payoffs()) == 0


def test_replay_reconstructs_terminal_state() -> None:
    h = new_hand([500, 500, 500], button=2, bb=100, sb=50,
                 hole=_hole(("Ac", "Ad"), ("Kc", "Kd"), ("Qc", "Qd")),
                 board=_board("2h", "7d", "8s", "3c", "4h"))
    h.apply(("raise", 300))  # seat2 button
    h.apply(("fold", 0))     # seat0 SB folds
    h.apply(("call", 300))   # seat1 BB calls -> flop with seat1, seat2
    while not h.is_terminal():  # check the rest of the way down
        h.apply(("check", 0))
    assert h.is_terminal()
    replayed = h.replay(h.setup, h.action_history)
    assert replayed.is_terminal() == h.is_terminal()
    assert replayed.final_stacks() == h.final_stacks()
    assert replayed.payoffs() == h.payoffs()
