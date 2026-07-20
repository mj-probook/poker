"""Slice F exit: a fixture session is 100% routed and splits exact/approx."""

import dataclasses
from pathlib import Path

import pytest

import pytest

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

# --------------------------------------------------------------------------- #
# Round-1 finding [8]: per-hand isolation. One unparseable/unreplayable hand
# must not abort the whole session — it is reported, the rest still grade.
# --------------------------------------------------------------------------- #
def _poisoned_hand():
    """A hand whose replay raises: the first action over-commits the stack."""
    ph = parse_pokerstars((FIXTURES / "ps_multiway_flop.txt").read_text())
    return dataclasses.replace(ph, actions=[(ph.actions[0][0], ("raise", 10**9))])

def test_one_bad_hand_does_not_abort_the_session() -> None:
    good = _session()
    clean = grade_session(good, population=load_population())

    session = good[:2] + [_poisoned_hand()] + good[2:]
    report = grade_session(session, population=load_population())

    # every good hand still graded
    assert report.total == clean.total
    # and the failure is reported, not swallowed
    assert len(report.failed_hands) == 1
    fh = report.failed_hands[0]
    assert fh.hand_index == 2
    assert fh.reason and "exceeds stack" in fh.reason
    assert report.failed_hand_count == 1

def test_clean_session_reports_no_failures() -> None:
    report = grade_session(_session(), population=load_population())
    assert report.failed_hands == []
    assert report.failed_hand_count == 0

def test_failed_hand_carries_source_hand_id() -> None:
    poisoned = _poisoned_hand()
    report = grade_session([poisoned], population=load_population())
    assert report.total == 0
    assert len(report.failed_hands) == 1
    assert report.failed_hands[0].hand_id == poisoned.hand_id


# --------------------------------------------------------------------------- #
# Round-1 findings [13][14]: the production path replayed hands without ever
# checking the replay against the history's own stated result, and applied
# actions without checking whose turn it was — so a desynced action list was
# silently MISATTRIBUTED to the wrong seat rather than rejected.
# --------------------------------------------------------------------------- #
def test_hand_whose_replay_contradicts_the_history_is_not_graded() -> None:
    """[13] If the engine and the HH disagree, the hand is untrustworthy."""
    good = _session()
    ph = parse_pokerstars((FIXTURES / "ps_multiway_flop.txt").read_text())
    # doctor the stated result: claim the hero collected 1 chip more
    collected = list(ph.collected)
    collected[ph.hero] += 1
    doctored = dataclasses.replace(ph, collected=tuple(collected))

    report = grade_session(good + [doctored], population=load_population())

    clean = grade_session(good, population=load_population())
    assert report.total == clean.total          # nothing from the bad hand
    assert report.failed_hand_count == 1
    assert "reconcil" in report.failed_hands[0].reason.lower()


def test_desynced_action_list_is_rejected_not_misattributed() -> None:
    """[14] An action for the wrong seat must raise, not silently reassign."""
    ph = parse_pokerstars((FIXTURES / "ps_multiway_flop.txt").read_text())
    n = len(ph.setup.stacks)
    first_seat, first_action = ph.actions[0]
    desynced = dataclasses.replace(
        ph, actions=[((first_seat + 1) % n, first_action)] + list(ph.actions))

    report = grade_session([desynced], population=load_population())

    assert report.total == 0
    assert report.failed_hand_count == 1
    assert "to_act" in report.failed_hands[0].reason


def test_seat_guard_raises_from_extract_decisions() -> None:
    """[14] The guard lives at the apply boundary, so every caller gets it."""
    from pokerlab.hh.decisions import extract_decisions

    ph = parse_pokerstars((FIXTURES / "ps_multiway_flop.txt").read_text())
    n = len(ph.setup.stacks)
    first_seat, first_action = ph.actions[0]
    desynced = dataclasses.replace(
        ph, actions=[((first_seat + 1) % n, first_action)] + list(ph.actions))

    with pytest.raises(ValueError, match="to_act"):
        extract_decisions(desynced)
