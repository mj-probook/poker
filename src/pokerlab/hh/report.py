"""Session-level HH grading report (Slice F exit; impl doc §3).

Routes every hero decision in a batch of parsed hands (exit: 100% routed) and
grades each by tier, then splits the results into:

  * exact  — ev_loss populated (tier 1 chart, tier 2 solved);
  * approx — tier 3 graded against the population table (ev_loss stays None);
  * pending — no reference yet (tier 2 awaiting a solve, or a preflop spot the
    chart does not model). Tier-2 pendings are what the batch worker drains.

This is pure (no store); `hh.persist` writes the same gradings + batch_queue
rows through Slice E's `store`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pokerlab.hh.decisions import Decision, extract_decisions, reconcile
from pokerlab.hh.grade import Grading, grade_decision
from pokerlab.hh.model import ParsedHand
from pokerlab.hh.population import Population
from pokerlab.types import Solution, TIER_SOLVER

SolutionLookup = Callable[[Decision], Solution | None]


@dataclass
class GradedDecision:
    hand_index: int          # index of the source hand in the session
    decision: Decision
    grading: Grading


@dataclass
class FailedHand:
    """A hand that could not be graded, and why (round-1 finding [8]).

    Real hand histories carry malformed and unreplayable hands; one of them
    must never cost the session. Failures are surfaced here rather than
    swallowed, so a session report is never silently short.
    """

    hand_index: int          # index of the source hand in the session
    hand_id: str | None      # site hand number, when the parse got that far
    reason: str


@dataclass
class SessionReport:
    graded: list[GradedDecision] = field(default_factory=list)
    failed_hands: list[FailedHand] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.graded)

    @property
    def failed_hand_count(self) -> int:
        return len(self.failed_hands)

    @property
    def routed(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for gd in self.graded:
            counts[gd.grading.tier] = counts.get(gd.grading.tier, 0) + 1
        return counts

    @property
    def exact(self) -> list[GradedDecision]:
        return [gd for gd in self.graded if gd.grading.ev_loss is not None]

    @property
    def approx(self) -> list[GradedDecision]:
        return [gd for gd in self.graded
                if gd.grading.ev_loss is None and gd.grading.graded]

    @property
    def pending(self) -> list[GradedDecision]:
        return [gd for gd in self.graded if not gd.grading.graded]

    @property
    def tier2_pending(self) -> list[GradedDecision]:
        return [gd for gd in self.pending if gd.grading.tier == TIER_SOLVER]


def grade_session(
    parsed_hands: list[ParsedHand],
    *,
    population: Population,
    solution_for: SolutionLookup | None = None,
) -> SessionReport:
    report = SessionReport()
    for hand_index, ph in enumerate(parsed_hands):
        hand_id = getattr(ph, "hand_id", None)
        # Per-hand boundary: replay/extraction is where malformed histories bite
        # (bad amounts, desynced action lists, seat mismatches). A hand that
        # cannot be replayed is reported and skipped, never fatal.
        try:
            # Reconcile before grading: a hand whose replay contradicts its own
            # stated result is not understood, so nothing it produces is
            # trustworthy (round-1 finding [13]).
            reconcile(ph)
            decisions = extract_decisions(ph)
        except Exception as exc:  # noqa: BLE001 - isolation boundary
            report.failed_hands.append(
                FailedHand(hand_index, hand_id, f"{type(exc).__name__}: {exc}"))
            continue
        # Per-decision boundary: a single unhandled spot loses that decision,
        # not the hand's other (correctly graded) decisions.
        for d in decisions:
            try:
                sol = (solution_for(d)
                       if solution_for is not None and d.tier == TIER_SOLVER
                       else None)
                g = grade_decision(d, population=population, solution=sol)
            except Exception as exc:  # noqa: BLE001 - isolation boundary
                report.failed_hands.append(FailedHand(
                    hand_index, hand_id,
                    f"decision {d.index}: {type(exc).__name__}: {exc}"))
                continue
            report.graded.append(GradedDecision(hand_index, d, g))
    return report
