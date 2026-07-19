"""Slice H: fixed action/obs encodings round-trip and stay legal."""

import copy

from pokerlab.engine import new_hand
from pokerlab.engine.cards import card_from_str as C
from pokerlab.env.encoding import (
    DEFAULT_FRACTIONS,
    OBS_DIM,
    decode_action,
    encode_obs,
    legal_action_mask,
    legal_engine_actions,
    num_actions,
)


def _hand():
    return new_hand([2000, 2000, 2000], button=2, bb=100, sb=50,
                    hole=[(C("As"), C("Ah")), (C("Kd"), C("Kc")), (C("Qs"), C("Qh"))],
                    board=[C("2c"), C("7d"), C("9h"), C("Js"), C("Ts")])


def test_action_space_width() -> None:
    assert num_actions() == 3 + len(DEFAULT_FRACTIONS)


def test_every_legal_decoded_action_is_accepted_by_engine() -> None:
    # Walk a few plies; at each, every masked index must apply cleanly.
    h = _hand()
    steps = 0
    while not h.is_terminal() and steps < 12:
        legal = legal_engine_actions(h)
        assert legal, "actor with no legal action"
        for idx, action in legal:
            probe = copy.deepcopy(h)
            probe.apply(action)  # must not raise
        # advance the real hand by the first legal action
        h.apply(legal[0][1])
        steps += 1


def test_preflop_mask_matches_expectation() -> None:
    h = _hand()  # UTG (button+3) faces the BB
    mask = legal_action_mask(h)
    names = ["fold", "check_call", "bet_0.5p", "bet_1.0p", "allin"]
    d = dict(zip(names, mask))
    assert d["fold"] and d["check_call"] and d["allin"]
    assert d["bet_0.5p"] and d["bet_1.0p"]  # deep stacks -> both sizes legal


def test_decode_allin_commits_whole_stack() -> None:
    h = _hand()
    seat = h.to_act
    label, amt = decode_action(h, num_actions() - 1)  # last index = all-in
    assert label == "allin"
    assert amt == h.street_bet[seat] + h.stack_left[seat]


def test_check_when_not_facing_a_bet() -> None:
    h = new_hand([2000, 2000], button=1, bb=100, sb=50, hole=[None, None])
    h.apply(("call", 100))  # SB completes -> BB may check
    assert decode_action(h, 1) == ("check", 0)


def test_obs_shape_and_determinism() -> None:
    h = _hand()
    o1, o2 = encode_obs(h), encode_obs(h)
    assert o1.shape == (OBS_DIM,)
    assert (o1 == o2).all()
    assert o1.any()  # hero hole + stacks populated
