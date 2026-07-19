"""Slice D — resolve_spot_key(state) -> SpotKey (impl doc §1, §3 Slice D).

Formation from preflop action history; stack bucket to the nearest of
10/20/40/100 bb; board bucket "iso:texture".
"""

from __future__ import annotations

import dataclasses

from pokerlab.engine import card_from_str, new_hand
from pokerlab.solver import iso, texture
from pokerlab.solver.spotkey import resolve_spot_key
from pokerlab.types import SpotKey, TournamentContext


def _flop(*strs: str) -> list[int]:
    return [card_from_str(s) for s in strs]


def _tc(bb: int) -> TournamentContext:
    return TournamentContext(payouts=(0,), players_remaining=2, stacks_all=(4000, 4000), bb=bb)


def _hu_btn_open_bb_call(stack: int = 4000, bb: int = 100):
    board = _flop("9h", "8h", "7c") + _flop("2d", "2s")  # flop + turn + river
    hole = [(card_from_str("Ah"), card_from_str("Kd")), (card_from_str("Qc"), card_from_str("Jc"))]
    hand = new_hand([stack, stack], button=1, bb=bb, sb=bb // 2, hole=hole, board=board)
    # HU: seat1 = BTN/SB acts first preflop; opens; seat0 = BB calls.
    hand.apply(("raise", int(2.5 * bb)))
    hand.apply(("call", int(2.5 * bb)))
    assert hand.game_state.street == "flop"
    return hand


def _attach_tc(state, bb):
    return dataclasses.replace(state, tournament=_tc(bb))


def test_returns_spotkey_type():
    st = _attach_tc(_hu_btn_open_bb_call().game_state, 100)
    key = resolve_spot_key(st)
    assert isinstance(key, SpotKey)


def test_board_bucket_is_iso_colon_texture():
    st = _attach_tc(_hu_btn_open_bb_call().game_state, 100)
    key = resolve_spot_key(st)
    flop = tuple(st.board[:3])
    assert key.board_bucket == f"{iso.iso_class(flop)}:{texture.texture(flop)}"
    assert key.board_bucket.endswith(":twotone_conn")


def test_stack_bucket_snaps_to_nearest_supported_depth():
    # 40bb exact
    st = _attach_tc(_hu_btn_open_bb_call(stack=4000, bb=100).game_state, 100)
    assert resolve_spot_key(st).stack_bucket == 40
    # 22bb -> nearest of {10,20,40,100} is 20
    st20 = _attach_tc(_hu_btn_open_bb_call(stack=2200, bb=100).game_state, 100)
    assert resolve_spot_key(st20).stack_bucket == 20
    # 90bb -> nearest is 100
    st100 = _attach_tc(_hu_btn_open_bb_call(stack=9000, bb=100).game_state, 100)
    assert resolve_spot_key(st100).stack_bucket == 100


def test_formation_heads_up_btn_open_bb_call():
    st = _attach_tc(_hu_btn_open_bb_call().game_state, 100)
    assert resolve_spot_key(st).formation == "BTNopen_BBcall"


def test_formation_three_handed_btn_open_bb_call():
    board = _flop("Ah", "Kd", "2c") + _flop("5s", "9d")
    hole = [
        (card_from_str("2h"), card_from_str("3h")),
        (card_from_str("Qc"), card_from_str("Jc")),
        (card_from_str("Ac"), card_from_str("Kc")),
    ]
    # seats: SB=0, BB=1, BTN=2 (button=2, n=3). BTN acts first preflop.
    hand = new_hand([4000, 4000, 4000], button=2, bb=100, sb=50, hole=hole, board=board)
    hand.apply(("raise", 250))  # BTN opens
    hand.apply(("fold", 0))  # SB folds
    hand.apply(("call", 250))  # BB calls
    st = _attach_tc(hand.game_state, 100)
    assert resolve_spot_key(st).formation == "BTNopen_BBcall"


def test_formation_ignores_postflop_actions():
    hand = _hu_btn_open_bb_call()
    # add a flop action; formation must stay a preflop property
    hand.apply(("check", 0))
    st = _attach_tc(hand.game_state, 100)
    assert resolve_spot_key(st).formation == "BTNopen_BBcall"
