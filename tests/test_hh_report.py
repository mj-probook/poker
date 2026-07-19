"""Slice F exit: a fixture session is 100% routed and splits exact/approx."""

from pathlib import Path

from pokerlab.hh.ggpoker import parse_ggpoker
from pokerlab.hh.pokerstars import parse_pokerstars
from pokerlab.hh.population import load_population
from pokerlab.hh.report import grade_session
from pokerlab.types import TIER_BEST_AVAILABLE, TIER_CHART, TIER_SOLVER

FIXTURES = Path(__file__).parent / "fixtures" / "hh"


def _session():
    ps = [parse_pokerstars((FIXTURES / f).read_text()) for f in (
        "ps_preflop_fold.txt", "ps_multiway_flop.txt",
        "ps_allin_sidepot.txt", "ps_ante_hu.txt")]
    gg = [parse_ggpoker((FIXTURES / f).read_text()) for f in (
        "gg_preflop_3bet.txt", "gg_allin_jam.txt", "gg_sb_fold_leak.txt")]
    return ps + gg


def test_every_decision_is_routed_to_a_tier() -> None:
    report = grade_session(_session(), population=load_population())
    assert report.total > 0
    # 100% routed: the per-tier counts sum to every graded decision.
    assert sum(report.routed.values()) == report.total
    assert set(report.routed) <= {TIER_CHART, TIER_SOLVER, TIER_BEST_AVAILABLE}


def test_report_splits_exact_and_approx() -> None:
    report = grade_session(_session(), population=load_population())
    # exact = ev_loss populated (tier 1 chart); approx = tier-3 population;
    # every bucket is disjoint and together with pending covers the whole set.
    assert len(report.exact) + len(report.approx) + len(report.pending) == report.total
    assert report.exact, "expected at least one chart-graded (exact) decision"
    # exact decisions must all carry an ev_loss; approx must all be ev_loss-free
    assert all(gd.grading.ev_loss is not None for gd in report.exact)
    assert all(gd.grading.ev_loss is None for gd in report.approx)
    # tier-3 (multiway postflop) decisions never leak an ev_loss
    for gd in report.graded:
        if gd.grading.tier == TIER_BEST_AVAILABLE:
            assert gd.grading.ev_loss is None
