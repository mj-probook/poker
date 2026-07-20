"""Hand evaluation (impl doc §3 Slice A).

Thin wrapper over ``phevaluator`` (a pure-python lookup evaluator; no C
extension required). Returns phevaluator's rank where **smaller is stronger**
(1 = royal flush, 7462 = 7-high). Card ids are the frozen `types.Card` codec,
which is identical to phevaluator's integer ids, so they pass through directly.
"""

from __future__ import annotations

from collections.abc import Sequence

from phevaluator import evaluate_cards

from pokerlab.types import Card

BEST_RANK = 1  # royal flush
WORST_RANK = 7462  # 7-5-4-3-2


def rank_hand(cards: Sequence[Card]) -> int:
    """Rank the best five-card hand from 5..7 cards. Smaller is stronger."""
    n = len(cards)
    if n < 5 or n > 7:
        raise ValueError(f"need 5..7 cards, got {n}")
    if len(set(cards)) != n:
        raise ValueError(f"duplicate cards: {list(cards)}")
    return evaluate_cards(*cards)


def rank_showdown(hole: tuple[Card, Card], board: Sequence[Card]) -> int:
    """Best-five rank for a hole pair against the (>=3 card) board."""
    return rank_hand((*hole, *board))
