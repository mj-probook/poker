"""M4 hand-history pipeline: parse → replay-validate → route → grade → persist.

    from pokerlab.hh import (
        parse_pokerstars, parse_ggpoker, ParsedHand,          # parsers
        extract_decisions, Decision, route_tier,              # routing
        grade_decision, grade_session, SessionReport,         # grading
        reconcile, FailedHand,                                 # replay validation / failure path
        load_population, Population,                           # tier-3 baseline
        persist_session, drain_batch_queue,                   # store (Slice E)
    )

A session can DROP hands: `reconcile` validates each replay against the HH's
stated results, and hands that fail parse/replay/reconciliation land in
`SessionReport.failed_hands` (as `FailedHand`) — never partially graded, never
claiming their dedup uid, so a parser fix re-imports them.

Parsers turn anonymized PokerStars/GGPoker tournament text into a `HandSetup` +
engine action list, validated by replaying through the Slice-A engine. Every
hero decision is routed to a grading tier (preflop/≤20bb → 1, HU postflop → 2,
multiway postflop → 3) and graded: tier 1 vs the chart engine (ev_loss), tier 3
vs the population table (ev_loss ALWAYS None — grading-honesty hard rule). The
`store` layer persists gradings + a tier-2 solve queue and serves leak reports.
"""

from pokerlab.hh.decisions import (
    Decision,
    extract_decisions,
    hand_label,
    position_label,
    reconcile,
)
from pokerlab.hh.ggpoker import parse_ggpoker
from pokerlab.hh.grade import (
    Grading,
    grade_decision,
    grade_tier1,
    grade_tier2,
    grade_tier3,
)
from pokerlab.hh.model import ParsedHand
from pokerlab.hh.persist import (
    drain_batch_queue,
    parse_by_site,
    persist_session,
    spot_key,
)
from pokerlab.hh.pokerstars import parse_pokerstars
from pokerlab.hh.population import Population, load_population
from pokerlab.hh.report import (
    FailedHand,
    GradedDecision,
    SessionReport,
    grade_session,
)
from pokerlab.hh.tiers import route_tier

__all__ = [
    "parse_pokerstars",
    "parse_ggpoker",
    "parse_by_site",
    "ParsedHand",
    "Decision",
    "extract_decisions",
    "hand_label",
    "position_label",
    "route_tier",
    "Grading",
    "grade_decision",
    "grade_tier1",
    "grade_tier2",
    "grade_tier3",
    "Population",
    "load_population",
    "SessionReport",
    "GradedDecision",
    "FailedHand",
    "grade_session",
    "reconcile",
    "persist_session",
    "drain_batch_queue",
    "spot_key",
]
