"""Slice D — suit-isomorphism canonicalization of flops (impl doc §3 Slice D).

A flop is a set of 3 board cards. Suit relabelling (any of the 24 permutations
of the 4 suits) does not change strategy, so the C(52,3)=22100 flops collapse to
1755 canonical classes. These tests pin the count, idempotence, orbit-invariance
of the canonical form, and the consistency of the induced hole-card map.
"""

from __future__ import annotations

import itertools
import random

import pytest

from pokerlab.engine.cards import card_from_str, card_rank, card_suit, make_card
from pokerlab.solver import iso


def _perm_card(card: int, perm: tuple[int, ...]) -> int:
    return (card // 4) * 4 + perm[card % 4]


ALL_PERMS = list(itertools.permutations(range(4)))


def test_exactly_1755_canonical_flops():
    canon = iso.all_canonical_flops()
    assert len(canon) == 1755
    assert len(set(canon)) == 1755  # all distinct


def test_every_flop_maps_into_the_canonical_set():
    canon = set(iso.all_canonical_flops())
    # Spot-check a deterministic sample of the 22100 flops (full sweep in the
    # count test path); every canonicalisation lands in the canonical set.
    rng = random.Random(7)
    deck = list(range(52))
    for _ in range(2000):
        flop = tuple(rng.sample(deck, 3))
        assert iso.canonical_flop(flop) in canon


def test_canonicalization_is_idempotent():
    for c in iso.all_canonical_flops()[::37]:
        assert iso.canonical_flop(c) == c


def test_isomorphic_flops_share_a_canonical_form():
    rng = random.Random(11)
    deck = list(range(52))
    for _ in range(500):
        flop = tuple(rng.sample(deck, 3))
        perm = rng.choice(ALL_PERMS)
        iso_flop = tuple(_perm_card(c, perm) for c in flop)
        assert iso.canonical_flop(iso_flop) == iso.canonical_flop(flop)


def test_canonical_flop_is_sorted_and_rank_preserving():
    flop = (card_from_str("Kh"), card_from_str("2c"), card_from_str("Ks"))
    canon = iso.canonical_flop(flop)
    assert list(canon) == sorted(canon)
    assert sorted(card_rank(c) for c in canon) == sorted(card_rank(c) for c in flop)


def test_canonical_combo_is_invariant_under_suit_permutation():
    # The induced hole-card map must be consistent: canonicalising (flop, hole)
    # and canonicalising the suit-permuted (flop, hole) yield identical output.
    rng = random.Random(23)
    deck = list(range(52))
    for _ in range(500):
        five = rng.sample(deck, 5)
        flop, hole = tuple(five[:3]), tuple(five[3:])
        perm = rng.choice(ALL_PERMS)
        f2 = tuple(_perm_card(c, perm) for c in flop)
        h2 = tuple(_perm_card(c, perm) for c in hole)
        assert iso.canonical_combo(flop, hole) == iso.canonical_combo(f2, h2)


def test_iso_class_string_stable_across_orbit():
    flop = (card_from_str("Ah"), card_from_str("Kd"), card_from_str("2c"))
    perm = (1, 2, 3, 0)
    f2 = tuple(_perm_card(c, perm) for c in flop)
    assert iso.iso_class(flop) == iso.iso_class(f2)
    # human-readable, 6 chars for a 3-card flop
    assert len(iso.iso_class(flop)) == 6
