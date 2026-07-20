"""Slice A: NLHE state machine — posting, legality, side pots, showdown, replay.

Outcomes here are hand-computed; the PokerKit differential (test_engine_diff)
is the exhaustive oracle.
"""

import pytest

from pokerlab.engine.cards import card_from_str as C
from pokerlab.engine.state import Hand, HandSetup, new_hand


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


# --------------------------------------------------------------------------- #
# Wave-2 [E1]: a betting round with nobody to bet against is CLOSED. The engine
# kept the last live player in the queue and demanded a ('check', 0) no-op —
# an action no real hand history records, so replaying a legitimate short-stack
# hand stalled short of terminal and hh.reconcile dropped it.
#
# The gap was preflop only: _advance_street already closed postflop streets
# whose can-act count fell below two.
# --------------------------------------------------------------------------- #
def test_preflop_closes_when_the_blind_puts_the_only_opponent_allin() -> None:
    """HU: BB has less than a big blind, so posting it puts them all-in.

    The SB then faces no outstanding wager and has nobody who can call, so
    there is no decision to make and the hand should run out.
    """
    h = new_hand([48, 5686], button=1, bb=100, sb=50,
                 hole=_hole(("Ah", "Kd"), ("Qc", "Jc")),
                 board=_board("2c", "7d", "9h", "Ts", "3s"))
    assert h.allin[0] and not h.allin[1]
    assert h.to_act is None, "no player has a decision to make here"
    assert h.is_terminal(), "the hand should run out, not demand a no-op check"


def test_preflop_still_asks_the_lone_player_who_owes_a_call() -> None:
    """The case a naive 'one player left -> close' guard breaks.

    Facing an all-in that exceeds their blind, the last live player owes a
    wager and MUST still get fold/call — closing here would silently fold them.
    """
    h = new_hand([5000, 3000], button=1, bb=100, sb=50,
                 hole=_hole(("Ah", "Kd"), ("Qc", "Jc")),
                 board=_board("2c", "7d", "9h", "Ts", "3s"))
    h.apply(("allin", 3000))          # SB/button jams
    assert h.to_act == 0, "the BB still has a real decision"
    assert not h.is_terminal()
    labels = {a[0] for a in h.legal_actions()}
    assert "fold" in labels and ("call" in labels or "allin" in labels)


def test_multiway_preflop_closes_when_only_one_can_act_and_owes_nothing() -> None:
    h = new_hand([870, 10, 7, 35], button=3, bb=100, sb=50,
                 hole=[None, None, None, None],
                 board=_board("2c", "7d", "9h", "Ts", "3s"))
    # every short stack is all-in from its blind/ante; nobody can be bet against
    if sum(not h.allin[i] for i in range(4)) <= 1:
        assert h.to_act is None or h.current_bet - h.street_bet[h.to_act] > 0


# --------------------------------------------------------------------------- #
# Wave-3 [A1] (P0): a big-blind ante is posted by ONE seat for the TABLE, so it
# is not that seat's stake. The pot ladder cut on total contribution, which put
# the ante in a top layer only the poster was eligible for — i.e. handed it back
# as an uncalled bet. A BB that posted the ante, went all-in and LOST still
# collected its own ante.
# --------------------------------------------------------------------------- #
def _ante_showdown(bb_ante: int, stacks=(150, 250, 150)):
    """3-handed all-in; seat1 is BB, posts the ante, and holds the WORST hand."""
    def rc(r, s): return (r - 2) * 4 + s
    hole = ((rc(14, 3), rc(14, 2)), (rc(7, 0), rc(3, 1)), (rc(13, 3), rc(13, 2)))
    board = (rc(12, 0), rc(9, 1), rc(5, 2), rc(2, 3), rc(8, 0))  # helps nobody
    setup = HandSetup(stacks=stacks, button=2, bb=100, sb=50, ante=0,
                      hole=hole, board=board, bb_ante=bb_ante)
    h = Hand(setup)
    while not h.is_terminal():
        acts = h.legal_actions()
        h.apply(next((a for a in acts if a[0] == "allin"), None)
                or next((a for a in acts if a[0] == "call"), acts[0]))
    return h


def test_a_losing_big_blind_does_not_get_its_ante_back():
    h = _ante_showdown(bb_ante=100)
    # 550 chips are in play; every live commitment is 150, so NOTHING is
    # uncalled. seat1's 100 ante is dead money and it holds the worst hand.
    assert sum(h.contrib) == 550
    assert h.final_stacks() == [550, 0, 0]


def test_a_table_ante_is_not_the_posters_stake():
    h = _ante_showdown(bb_ante=100)
    # The ante rides in `contrib` (it is real money in the pot) but is excluded
    # from the poster's stake, which is what the ladder and the auto-muck test
    # are computed on.
    assert h.contrib[1] == 250
    assert h.table_dead == [0, 100, 0]


def test_chips_are_conserved_with_a_table_ante():
    for bb_ante, stacks in [(100, (150, 250, 150)), (200, (70, 500, 300)),
                            (500, (90, 120, 3000))]:
        h = _ante_showdown(bb_ante=bb_ante, stacks=stacks)
        assert sum(h.final_stacks()) == sum(stacks), (
            f"bb_ante={bb_ante} stacks={stacks}: pot does not add up")


def test_an_uncalled_bet_still_comes_back_with_a_table_ante_in_play():
    """The fix must not overshoot: genuinely unmatched LIVE chips still return.

    Everyone folds to a BB that posted the ante, so the BB's own blind is
    uncalled — it comes back — and the BB also collects the dead ante and the
    blinds, because it is the only player left.
    """
    def rc(r, s): return (r - 2) * 4 + s
    hole = ((rc(14, 3), rc(14, 2)), (rc(7, 0), rc(3, 1)), (rc(13, 3), rc(13, 2)))
    board = (rc(12, 0), rc(9, 1), rc(5, 2), rc(2, 3), rc(8, 0))
    setup = HandSetup(stacks=(1000, 1000, 1000), button=2, bb=100, sb=50,
                      ante=0, hole=hole, board=board, bb_ante=100)
    h = Hand(setup)
    while not h.is_terminal():
        acts = h.legal_actions()
        h.apply(next((a for a in acts if a[0] == "fold"), acts[0]))
    assert sum(h.final_stacks()) == 3000
    assert h.final_stacks()[1] == 1050          # BB wins the SB, ante is its own
