"""M0 engine: cards, hand evaluation, NLHE state machine, tournament helpers.

Public surface for downstream slices (plan §3b; impl doc §1). The engine operates on a
mutable `Hand` state machine; `Hand.game_state` snapshots into the frozen
`types.GameState` contract that scoring/grading/solvers consume.
"""

from pokerlab.engine.cards import (
    Deck,
    card_from_str,
    card_rank,
    card_suit,
    card_to_str,
    cards_to_str,
    full_deck,
    make_card,
)
from pokerlab.engine.evaluator import rank_hand, rank_showdown
from pokerlab.engine.state import Hand, HandSetup, new_hand
from pokerlab.engine.tournament import (
    BlindLevel,
    effective_bb,
    effective_bb_all,
    is_monotone_schedule,
    m_ratio,
)

__all__ = [
    "Deck",
    "make_card",
    "card_rank",
    "card_suit",
    "card_to_str",
    "card_from_str",
    "cards_to_str",
    "full_deck",
    "rank_hand",
    "rank_showdown",
    "Hand",
    "HandSetup",
    "new_hand",
    "effective_bb",
    "effective_bb_all",
    "m_ratio",
    "BlindLevel",
    "is_monotone_schedule",
]
