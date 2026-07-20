"""Fixed observation/action encodings for the vectorized env (impl doc §1, §3 H).

This is the Rust-port seam: the encodings here are the contract a future native
env must reproduce bit-for-bit. Kept deliberately small and deterministic.

Action space (discrete, fixed width): ``fold``, ``check_call``, one ``bet_{f}p``
per pot-fraction in ``fractions`` (raise-to = current bet + f·pot, clamped to the
legal raise band), and ``allin``. ``legal_action_mask`` marks which indices the
engine will accept; ``decode_action`` turns an index into a concrete engine
`Action`. Observation is a fixed-width float vector of the acting seat's view.
"""

from __future__ import annotations

import numpy as np

from pokerlab.engine.cards import card_rank, card_suit
from pokerlab.engine.state import Hand
from pokerlab.types import Action

MAX_SEATS = 9
DEFAULT_FRACTIONS: tuple[float, ...] = (0.5, 1.0)


def action_names(fractions: tuple[float, ...] = DEFAULT_FRACTIONS) -> list[str]:
    return ["fold", "check_call", *[f"bet_{f}p" for f in fractions], "allin"]


def num_actions(fractions: tuple[float, ...] = DEFAULT_FRACTIONS) -> int:
    return 3 + len(fractions)


def _bet_target(hand: Hand, f: float) -> int | None:
    """Clamped raise-to for a pot-fraction bet, or None if no raise is legal."""
    bounds = hand.raise_bounds()
    if bounds is None:
        return None
    min_to, max_to = bounds
    raw = hand.current_bet + int(round(f * hand.pot))
    return min(max(raw, min_to), max_to)


def legal_action_mask(
    hand: Hand, fractions: tuple[float, ...] = DEFAULT_FRACTIONS
) -> np.ndarray:
    """Boolean mask over the discrete action indices for the acting seat."""
    mask = np.zeros(num_actions(fractions), dtype=bool)
    seat = hand.to_act
    if seat is None:
        return mask
    to_call = hand.current_bet - hand.street_bet[seat]
    mask[0] = to_call > 0  # fold only when facing a bet
    mask[1] = True  # check (to_call==0) or call is always available
    bounds = hand.raise_bounds()
    if bounds is not None:
        min_to, max_to = bounds
        for i, f in enumerate(fractions):
            t = _bet_target(hand, f)
            if t is not None and min_to <= t < max_to:  # distinct from all-in
                mask[2 + i] = True
        mask[2 + len(fractions)] = True  # all-in raise
    return mask


def decode_action(
    hand: Hand, idx: int, fractions: tuple[float, ...] = DEFAULT_FRACTIONS
) -> Action:
    """Turn a discrete action index into a concrete legal engine `Action`."""
    seat = hand.to_act
    if seat is None:
        raise ValueError("no actor")
    to_call = hand.current_bet - hand.street_bet[seat]
    max_to = hand.street_bet[seat] + hand.stack_left[seat]
    n_bets = len(fractions)
    if idx == 0:
        return ("fold", 0)
    if idx == 1:
        if to_call == 0:
            return ("check", 0)
        call_to = min(hand.current_bet, max_to)
        return ("allin" if call_to == max_to else "call", call_to)
    if idx == 2 + n_bets:  # all-in
        return ("allin", max_to)
    t = _bet_target(hand, fractions[idx - 2])
    if t is None:
        raise ValueError(f"action {idx} not legal here")
    label = "allin" if t == max_to else ("raise" if hand.current_bet > 0 else "bet")
    return (label, t)


def legal_engine_actions(
    hand: Hand, fractions: tuple[float, ...] = DEFAULT_FRACTIONS
) -> list[tuple[int, Action]]:
    """(index, engine action) for every currently-legal discrete action."""
    mask = legal_action_mask(hand, fractions)
    return [(i, decode_action(hand, i, fractions)) for i in range(len(mask)) if mask[i]]


# --- observation ----------------------------------------------------------- #
# Layout (all floats): hero hole (2*[rank/14, suit/3]) = 4; board (5*[rank/14,
# suit/3, present]) = 15; pot/bb = 1; per-seat (stack_bb, streetbet_bb, folded,
# allin, is_actor, is_button) for MAX_SEATS = 6*9 = 54; street one-hot = 4;
# to_call/bb = 1; min_raise/bb = 1.  Total:
OBS_DIM = 4 + 15 + 1 + 6 * MAX_SEATS + 4 + 1 + 1
_STREETS = ("preflop", "flop", "turn", "river", "showdown")


def encode_obs(hand: Hand) -> np.ndarray:
    """Fixed-width float view of the acting seat's information set."""
    v = np.zeros(OBS_DIM, dtype=np.float32)
    seat = hand.to_act
    if seat is None:
        return v
    bb = float(hand.bb)
    o = 0
    hole = hand.hole[seat]
    if hole is not None:
        for c in hole:
            v[o] = card_rank(c) / 14.0
            v[o + 1] = card_suit(c) / 3.0
            o += 2
    o = 4
    board = hand.full_board[: {"preflop": 0, "flop": 3, "turn": 4, "river": 5,
                               "showdown": 5}[_STREETS[hand.street_idx]]]
    for k in range(5):
        if k < len(board):
            v[o] = card_rank(board[k]) / 14.0
            v[o + 1] = card_suit(board[k]) / 3.0
            v[o + 2] = 1.0
        o += 3
    v[o] = hand.pot / bb
    o += 1
    for s in range(hand.n):
        base = o + 6 * s
        v[base] = hand.stack_left[s] / bb
        v[base + 1] = hand.street_bet[s] / bb
        v[base + 2] = float(hand.folded[s])
        v[base + 3] = float(hand.allin[s])
        v[base + 4] = float(s == seat)
        v[base + 5] = float(s == hand.button)
    o += 6 * MAX_SEATS
    v[o + hand.street_idx] = 1.0
    o += 4
    v[o] = (hand.current_bet - hand.street_bet[seat]) / bb
    v[o + 1] = hand.min_raise_inc / bb
    return v
