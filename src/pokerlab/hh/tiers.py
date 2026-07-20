"""Grading-tier router (Slice F, impl doc §3 / §5.3).

Table-driven so the routing policy is data, not branches: each rule is a
``(name, predicate, tier)`` triple evaluated top-to-bottom; the first match
wins. Policy (plan §5.3):

    preflop (incl. <=20bb jam/fold)  -> tier 1  (chart engine, exact)
    heads-up postflop                -> tier 2  (solution library)
    multiway postflop                -> tier 3  (best-available, ev_loss None)

`num_in_pot` is the count of non-folded players at the moment of the decision
(2 == heads-up). The <=20bb jam/fold spot is a preflop decision, so it already
routes to tier 1; it is called out separately only because it is the canonical
tier-1 *gradable* case (see `hh.grade`).
"""

from __future__ import annotations

from collections.abc import Callable

from pokerlab.types import TIER_BEST_AVAILABLE, TIER_CHART, TIER_SOLVER

# (name, predicate over (street, num_in_pot), tier)
ROUTING_RULES: list[tuple[str, Callable[[str, int], bool], int]] = [
    ("preflop", lambda street, n: street == "preflop", TIER_CHART),
    ("hu_postflop", lambda street, n: n == 2, TIER_SOLVER),
    ("multiway_postflop", lambda street, n: n >= 3, TIER_BEST_AVAILABLE),
]


# The ONE place a tier's honesty claim is worded. Plan §9 makes these
# load-bearing UI, "not fine print": every surface that shows a graded number
# has to say which oracle produced it, and tier 3 has to say it has none. One
# home so the drill page, the report page and the API cannot drift apart.
TIER_LABELS: dict[int, str] = {
    TIER_CHART: "exact — chart-graded",
    TIER_SOLVER: "exact — solver-graded",
    TIER_BEST_AVAILABLE: "approximate — population-graded, no EV loss",
}


def route_tier(street: str, num_in_pot: int) -> int:
    """Return the grading tier for a decision on ``street`` with ``num_in_pot``."""
    for _name, pred, tier in ROUTING_RULES:
        if pred(street, num_in_pot):
            return tier
    raise ValueError(f"unroutable decision: street={street!r} num_in_pot={num_in_pot}")
