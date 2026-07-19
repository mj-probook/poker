"""Card codec + seeded deck (plan §3b; impl doc §3 Slice A).

Card encoding is the frozen `types.Card` contract: ``card = (rank-2)*4 + suit``
with ``rank in 2..14`` (14 = ace) and ``suit in 0..3`` mapped
``0=c, 1=d, 2=h, 3=s``. This is byte-identical to phevaluator's integer card
id (verified), so evaluator calls pass these ints straight through.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from pokerlab.types import Card, Rank, Suit

RANKS: tuple[int, ...] = tuple(range(2, 15))  # 2..14 (14 = ace)
SUITS: tuple[int, ...] = (0, 1, 2, 3)

_RANK_CHARS = "23456789TJQKA"  # index 0 -> rank 2
_SUIT_CHARS = "cdhs"  # 0=c, 1=d, 2=h, 3=s (matches phevaluator)
_RANK_FROM_CHAR = {c: i + 2 for i, c in enumerate(_RANK_CHARS)}
_SUIT_FROM_CHAR = {c: i for i, c in enumerate(_SUIT_CHARS)}


def make_card(rank: Rank, suit: Suit) -> Card:
    """Build a card id from ``rank`` (2..14) and ``suit`` (0..3)."""
    if rank not in RANKS:
        raise ValueError(f"rank out of range: {rank}")
    if suit not in SUITS:
        raise ValueError(f"suit out of range: {suit}")
    return (rank - 2) * 4 + suit


def card_rank(card: Card) -> Rank:
    """Decode the rank (2..14) of a card id."""
    return card // 4 + 2


def card_suit(card: Card) -> Suit:
    """Decode the suit (0..3) of a card id."""
    return card % 4


def card_to_str(card: Card) -> str:
    """Two-char string, e.g. card for the ace of spades -> ``"As"``."""
    if not 0 <= card <= 51:
        raise ValueError(f"card id out of range: {card}")
    return _RANK_CHARS[card_rank(card) - 2] + _SUIT_CHARS[card_suit(card)]


def card_from_str(s: str) -> Card:
    """Parse a two-char card string like ``"As"`` (rank case-insensitive)."""
    if len(s) != 2:
        raise ValueError(f"card string must be 2 chars: {s!r}")
    rank_c, suit_c = s[0].upper(), s[1].lower()
    if rank_c not in _RANK_FROM_CHAR or suit_c not in _SUIT_FROM_CHAR:
        raise ValueError(f"bad card string: {s!r}")
    return make_card(_RANK_FROM_CHAR[rank_c], _SUIT_FROM_CHAR[suit_c])


def full_deck() -> list[Card]:
    """The 52 card ids in canonical (rank-major) order."""
    return list(range(52))


class Deck:
    """Deterministic seeded deck. Same seed -> same deal order, always."""

    def __init__(self, seed: int) -> None:
        self._cards = full_deck()
        random.Random(seed).shuffle(self._cards)
        self._pos = 0

    def deal(self, n: int) -> list[Card]:
        """Deal the next ``n`` cards off the top."""
        if n < 0 or self._pos + n > len(self._cards):
            raise ValueError(f"cannot deal {n} cards (remaining {self.remaining})")
        out = self._cards[self._pos : self._pos + n]
        self._pos += n
        return out

    @property
    def remaining(self) -> int:
        return len(self._cards) - self._pos


def cards_to_str(cards: Sequence[Card]) -> str:
    """Concatenate cards into a phevaluator/pokerkit-style string."""
    return "".join(card_to_str(c) for c in cards)
