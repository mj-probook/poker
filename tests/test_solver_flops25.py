"""Slice D — the checked-in 25-flop benchmark fixture (impl doc §3 Slice D).

Tests LOAD the fixture (never regenerate it) and pin its shape: 25 canonical
flops stratified across the 8 texture classes, correct labels, and the spot
parameters (ranges, pot, stack, bet grid).
"""

from __future__ import annotations

from collections import Counter

from pokerlab.solver import iso, texture
from pokerlab.solver.flops25 import build_solver, load_flops25, range_vectors


def test_fixture_has_25_flops():
    fx = load_flops25()
    assert len(fx["flops"]) == 25


def test_flops_are_canonical_and_correctly_labelled():
    fx = load_flops25()
    for entry in fx["flops"]:
        from pokerlab.engine.cards import card_from_str
        cards = tuple(card_from_str(entry["cards"][i:i + 2]) for i in range(0, 6, 2))
        assert iso.canonical_flop(cards) == cards, f"{entry['cards']} not canonical"
        assert entry["texture"] == texture.texture(cards)
        assert entry["iso_class"] == iso.iso_class(cards)


def test_flops_are_stratified_over_all_eight_classes():
    fx = load_flops25()
    dist = Counter(e["texture"] for e in fx["flops"])
    assert set(dist) == set(texture.TEXTURES)  # all 8 present
    assert min(dist.values()) >= 3  # ≥3 per class


def test_spot_parameters_pinned():
    meta = load_flops25()["meta"]
    assert meta["formation"] == "BTNopen_BBcall"
    assert meta["stack_bb"] == 40.0
    assert meta["pot_bb"] == 5.0
    assert meta["bet_grid"] == [0.33, 0.75, 1.25, "jam"]


def test_ranges_present_and_nonempty():
    fx = load_flops25()
    assert len(fx["ranges"]["BTN"]) > 20
    assert len(fx["ranges"]["BB"]) > 20
    bb, btn = range_vectors(fx)
    assert bb.sum() > 0 and btn.sum() > 0


def test_build_solver_masks_board_and_builds_tree():
    # A full-grid flop tree is a bench-scale object; here we only verify the
    # loader wires ranges + board + tree correctly (a reduced grid keeps it fast;
    # solving flops25 to the accuracy bar lives behind `make bench`).
    from pokerlab.solver.subgame import BetConfig, board_mask

    fx = load_flops25()
    cfg = BetConfig(sizes=(0.75,), jam=False, max_raises=0)
    s = build_solver(fx["flops"][0], cfg=cfg)
    # ranges masked by the board -> no combo collides with the flop
    assert (s.range0 * ~board_mask(s.board)).sum() == 0
    assert (s.range1 * ~board_mask(s.board)).sum() == 0
    assert len(s.decisions) > 0
    assert s.pot0 == 5.0
