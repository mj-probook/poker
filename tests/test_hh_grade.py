"""Slice F: tier-1 (chart) and tier-3 (population) grading.

Tier-1 populates ev_loss against the jam/fold chart via the shared decision-ε
rule; tier-3 NEVER reports ev_loss (grading-honesty hard rule) and instead
emits a frequency-deviation flag vs the population table.
"""

from pathlib import Path

from pokerlab.hh.decisions import extract_decisions
from pokerlab.hh.ggpoker import parse_ggpoker
from pokerlab.hh.grade import grade_decision, grade_tier1
from pokerlab.hh.pokerstars import parse_pokerstars
from pokerlab.hh.population import load_population
from pokerlab.types import TIER_BEST_AVAILABLE, TIER_CHART

FIXTURES = Path(__file__).parent / "fixtures" / "hh"
POP = load_population()


def _hero_decisions(parser, name):
    return extract_decisions(parser((FIXTURES / name).read_text()))


def test_tier1_correct_jam_has_zero_ev_loss() -> None:
    d = _hero_decisions(parse_ggpoker, "gg_allin_jam.txt")[0]
    g = grade_tier1(d)
    assert g.graded and g.tier == TIER_CHART
    assert g.chosen == "jam" and g.best == "jam"
    assert g.correct is True
    assert g.ev_loss is not None and g.ev_loss < 1e-6


def test_tier1_folding_a_jam_hand_is_a_leak() -> None:
    d = _hero_decisions(parse_ggpoker, "gg_sb_fold_leak.txt")[0]
    assert d.tier == TIER_CHART and d.position == "SB" and round(d.eff_bb) == 10
    g = grade_tier1(d)
    assert g.graded and g.chosen == "fold" and g.best == "jam"
    assert g.correct is False
    assert g.ev_loss is not None and g.ev_loss > 1.0  # ~1.14bb conceded


def test_preflop_non_jamfold_spot_is_ungraded() -> None:
    # Hero (BB) folds AK facing a raise (opponent not all-in): no chart model.
    d = _hero_decisions(parse_pokerstars, "ps_preflop_fold.txt")[0]
    g = grade_tier1(d)
    assert g.tier == TIER_CHART and g.graded is False and g.ev_loss is None


def test_tier3_multiway_never_reports_ev_loss() -> None:
    # Fabricate a multiway postflop decision by grading the flop bet in the
    # multiway fixture through the tier-3 grader.
    decisions = _hero_decisions(parse_pokerstars, "ps_multiway_flop.txt")
    flop_bet = decisions[1]
    assert flop_bet.tier == TIER_BEST_AVAILABLE
    g = grade_decision(flop_bet, population=POP)
    assert g.tier == TIER_BEST_AVAILABLE
    assert g.ev_loss is None            # honesty rule
    assert g.correct is None


def test_tier3_flags_rare_population_action() -> None:
    decisions = _hero_decisions(parse_pokerstars, "ps_multiway_flop.txt")
    flop_bet = decisions[1]  # formation "6max:HJ" — unknown spot -> no baseline
    # Force a known spot + a rare action by checking the population helper path.
    from pokerlab.hh.grade import grade_tier3
    from pokerlab.hh.decisions import Decision

    rare = Decision(
        index=0, street="flop", seat=0, position="BB", num_in_pot=3,
        pot=100, pot_bb=10.0, eff_bb=80.0, ante_bb=0.0, to_call=0,
        opp_allin=False, hole=(50, 44), board=(1, 2, 3),
        legal=[("check", 0)], chosen=("raise", 300), is_allin=False,
        tier=TIER_BEST_AVAILABLE, formation="6max:BB", action_type="raise",
    )
    g = grade_tier3(rare, POP)
    assert g.ev_loss is None
    assert "freq_deviation" in g.flags   # population raises here only 3%
    assert g.frequency is not None and g.frequency < 0.10
