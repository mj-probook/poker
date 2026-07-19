"""Slice D — 8-way flop texture labeler (impl doc §3 Slice D).

Total and deterministic: every one of the 1755 canonical flops gets exactly one
of 8 documented labels. Labels are suit-relabelling invariant.
"""

from __future__ import annotations

import itertools

from pokerlab.engine.cards import card_from_str
from pokerlab.solver import iso, texture


def _flop(*strs: str) -> tuple[int, ...]:
    return tuple(card_from_str(s) for s in strs)


def _perm_card(card: int, perm: tuple[int, ...]) -> int:
    return (card // 4) * 4 + perm[card % 4]


def test_there_are_exactly_8_labels():
    assert len(texture.TEXTURES) == 8
    assert len(set(texture.TEXTURES)) == 8


def test_known_boards_get_expected_labels():
    assert texture.texture(_flop("9h", "8h", "7c")) == "twotone_conn"
    assert texture.texture(_flop("Ah", "Kd", "2c")) == "rainbow_dry"
    assert texture.texture(_flop("As", "Ks", "Qs")) == "mono_conn"
    assert texture.texture(_flop("Ah", "7d", "2c")) == "rainbow_dry"
    assert texture.texture(_flop("Kh", "Ks", "5s")) == "paired_twotone"  # 5 shares a spade
    assert texture.texture(_flop("Kh", "Ks", "5c")) == "paired_rainbow"  # three suits
    assert texture.texture(_flop("Kh", "Kd", "Kc")) == "paired_rainbow"  # trips
    assert texture.texture(_flop("2c", "2d", "9h")) == "paired_rainbow"
    assert texture.texture(_flop("Ac", "2c", "3d")) == "twotone_conn"  # wheel-connected


def test_label_is_suit_permutation_invariant():
    flop = _flop("Ah", "Th", "6c")
    for perm in itertools.permutations(range(4)):
        p2 = tuple(_perm_card(c, perm) for c in flop)
        assert texture.texture(p2) == texture.texture(flop)


def test_every_canonical_flop_gets_exactly_one_known_label():
    labels = [texture.texture(f) for f in iso.all_canonical_flops()]
    assert len(labels) == 1755
    assert set(labels) <= set(texture.TEXTURES)


def test_all_eight_labels_are_populated_and_partition_1755():
    from collections import Counter

    counts = Counter(texture.texture(f) for f in iso.all_canonical_flops())
    assert sum(counts.values()) == 1755
    for label in texture.TEXTURES:
        assert counts[label] > 0, f"{label} is empty"
