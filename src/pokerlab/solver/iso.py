"""Suit-isomorphism canonicalization of flops (plan §1 SpotKey; impl doc §3 Slice D).

Relabelling the four suits never changes strategy, so the C(52,3)=22100 flops
collapse into **1755** canonical classes under the 24 suit permutations. This
module maps any flop to its canonical representative and canonicalises a
(flop, hole) combo consistently, so a solve computed on the canonical board can
be reused for every isomorphic board and range.

Cards use the frozen `types.Card` codec: ``card = (rank-2)*4 + suit`` — suit
relabelling acts only on the low 2 bits, ``card // 4`` (rank) is invariant.

Canonicalisation is *minimise the sorted card tuple over all 24 suit
permutations*. For combos we minimise ``(sorted_flop, sorted_hole)``
lexicographically, so the canonical flop is the primary key and the hole is
resolved within the flop's suit-automorphism coset — this makes the induced
hole map an orbit invariant (see test_canonical_combo_is_invariant...).
"""

from __future__ import annotations

import itertools
from functools import lru_cache

from pokerlab.engine.cards import card_to_str
from pokerlab.types import Card

_S4: tuple[tuple[int, ...], ...] = tuple(itertools.permutations(range(4)))


def _apply(perm: tuple[int, ...], card: Card) -> Card:
    return (card // 4) * 4 + perm[card % 4]


def canonical_flop(flop: tuple[Card, ...]) -> tuple[Card, ...]:
    """The canonical (suit-minimised, sorted) representative of ``flop``."""
    return min(tuple(sorted(_apply(p, c) for c in flop)) for p in _S4)


def canonical_combo(
    flop: tuple[Card, ...], hole: tuple[Card, ...]
) -> tuple[tuple[Card, ...], tuple[Card, ...]]:
    """Canonical ``(flop, hole)`` — flop-primary, hole resolved in the stabiliser.

    Invariant across the whole suit-permutation orbit of ``(flop, hole)``:
    strategically identical combos (e.g. two backdoor flush draws in
    interchangeable suits on a paired board) collapse to one canonical combo.
    """
    best: tuple[tuple[Card, ...], tuple[Card, ...]] | None = None
    for p in _S4:
        cf = tuple(sorted(_apply(p, c) for c in flop))
        ch = tuple(sorted(_apply(p, c) for c in hole))
        cand = (cf, ch)
        if best is None or cand < best:
            best = cand
    assert best is not None
    return best


def iso_class(flop: tuple[Card, ...]) -> str:
    """Compact human-readable id of the canonical flop, e.g. ``"AcKc2d"``."""
    return "".join(card_to_str(c) for c in canonical_flop(flop))


@lru_cache(maxsize=1)
def all_canonical_flops() -> tuple[tuple[Card, ...], ...]:
    """All 1755 canonical flops, ascending. Cached — full 22100-flop sweep."""
    seen: set[tuple[Card, ...]] = set()
    for flop in itertools.combinations(range(52), 3):
        seen.add(canonical_flop(flop))
    return tuple(sorted(seen))
