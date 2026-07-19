"""Slice F: table-driven tier router + hero decision extraction."""

from pathlib import Path

from pokerlab.hh.decisions import extract_decisions, hand_label, position_label
from pokerlab.hh.ggpoker import parse_ggpoker
from pokerlab.hh.pokerstars import parse_pokerstars
from pokerlab.hh.tiers import route_tier
from pokerlab.types import TIER_BEST_AVAILABLE, TIER_CHART, TIER_SOLVER

FIXTURES = Path(__file__).parent / "fixtures" / "hh"


def test_route_tier_table() -> None:
    assert route_tier("preflop", 2) == TIER_CHART
    assert route_tier("preflop", 6) == TIER_CHART
    assert route_tier("flop", 2) == TIER_SOLVER      # HU postflop
    assert route_tier("river", 2) == TIER_SOLVER
    assert route_tier("flop", 3) == TIER_BEST_AVAILABLE   # multiway postflop
    assert route_tier("turn", 5) == TIER_BEST_AVAILABLE


def test_hand_label() -> None:
    from pokerlab.engine.cards import card_from_str as C
    assert hand_label((C("Ah"), C("Kd"))) == "AKo"
    assert hand_label((C("Ah"), C("Ks"))) == "AKo"
    assert hand_label((C("Ah"), C("Kh"))) == "AKs"
    assert hand_label((C("Ac"), C("Ad"))) == "AA"
    assert hand_label((C("2c"), C("7d"))) == "72o"


def test_position_label_heads_up_and_six_max() -> None:
    assert position_label(2, 1, 1) == "SB"  # HU button is SB
    assert position_label(2, 1, 0) == "BB"
    assert position_label(6, 5, 0) == "SB"
    assert position_label(6, 5, 1) == "BB"
    assert position_label(6, 5, 5) == "BTN"


def test_multiway_flop_hand_routes_across_all_tiers() -> None:
    parsed = parse_pokerstars((FIXTURES / "ps_multiway_flop.txt").read_text())
    decisions = extract_decisions(parsed)
    # Hero: preflop raise (t1), flop bet 3-way (t3), turn bet HU (t2),
    # river check (t2), river call (t2).
    assert [d.street for d in decisions] == [
        "preflop", "flop", "turn", "river", "river"]
    assert [d.tier for d in decisions] == [
        TIER_CHART, TIER_BEST_AVAILABLE, TIER_SOLVER, TIER_SOLVER, TIER_SOLVER]
    assert decisions[1].num_in_pot == 3 and decisions[2].num_in_pot == 2


def test_jam_decision_is_preflop_tier1_allin() -> None:
    parsed = parse_ggpoker((FIXTURES / "gg_allin_jam.txt").read_text())
    decisions = extract_decisions(parsed)
    assert len(decisions) == 1
    d = decisions[0]
    assert d.tier == TIER_CHART and d.position == "SB"
    assert d.is_allin and d.action_type == "jam"
    assert round(d.eff_bb) == 15
    assert d.leak_key == "2max:SB|preflop|jam"
