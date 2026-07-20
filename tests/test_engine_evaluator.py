"""Slice A: hand evaluation ranks known hands, incl. wheel/steel-wheel edges."""

import pytest

from pokerlab.engine.cards import card_from_str
from pokerlab.engine.evaluator import rank_hand, rank_showdown


def _r(*cs: str) -> int:
    return rank_hand([card_from_str(c) for c in cs])


# Strictly descending strength ladder (smaller rank == stronger hand).
LADDER = [
    ("royal_flush", ("As", "Ks", "Qs", "Js", "Ts")),
    ("steel_wheel", ("As", "2s", "3s", "4s", "5s")),  # lowest straight flush
    ("quads", ("Ah", "Ad", "Ac", "As", "Kd")),
    ("full_house", ("Ah", "Ad", "Ac", "Kd", "Ks")),
    ("flush", ("As", "Js", "9s", "5s", "2s")),
    ("straight_6high", ("2c", "3d", "4h", "5s", "6c")),
    ("wheel_straight", ("Ac", "2d", "3h", "4s", "5c")),  # weakest straight
    ("trips", ("Ah", "Ad", "Ac", "Kd", "Qs")),
    ("two_pair", ("Ah", "Ad", "Kc", "Kd", "Qs")),
    ("pair", ("Ah", "Ad", "Kc", "Qd", "Js")),
    ("high_card", ("2c", "3d", "4h", "5s", "7c")),
]


def test_ladder_strictly_ordered() -> None:
    ranks = [_r(*cards) for _, cards in LADDER]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ranks)  # all distinct classes


def test_steel_wheel_beats_quads() -> None:
    # A-2-3-4-5 suited is a straight flush and must beat four of a kind.
    assert _r("As", "2s", "3s", "4s", "5s") < _r("Ah", "Ad", "Ac", "As", "Kd")


def test_wheel_straight_is_weakest_straight() -> None:
    assert _r("Ac", "2d", "3h", "4s", "5c") > _r("2c", "3d", "4h", "5s", "6c")


def test_best_five_of_seven() -> None:
    # Extra junk cards must not degrade a made royal flush.
    seven = ["As", "Ks", "Qs", "Js", "Ts", "2c", "3d"]
    assert rank_hand([card_from_str(c) for c in seven]) == 1


def test_rank_showdown_picks_best() -> None:
    hole = (card_from_str("As"), card_from_str("Ks"))
    board = [card_from_str(c) for c in ("Qs", "Js", "Ts", "2c", "3d")]
    assert rank_showdown(hole, board) == 1  # royal on board+hole


def test_rejects_dupes_and_bad_count() -> None:
    with pytest.raises(ValueError):
        _r("As", "As", "Ks", "Qs", "Js")
    with pytest.raises(ValueError):
        _r("As", "Ks", "Qs", "Js")  # only 4
