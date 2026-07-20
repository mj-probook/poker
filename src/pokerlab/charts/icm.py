"""Malmuth-Harville ICM (impl doc §3 Slice C; plan §4 "ICM").

The Independent Chip Model turns a stack vector into $ equity by assuming the
probability a player finishes first equals their share of chips in play, applied
recursively to award each prize place. No vendor data — pure combinatorics, the
Goal-A exercise the plan wants (§2 build item 2).

    icm_equities(stacks, payouts) -> list[float]   # $ equity per player

`stacks` are chips (any positive unit); `payouts` is the prize ladder in any
unit (the returned equities are in the payout unit). Places paid = min(#payouts,
#players); extra payouts (more places than players) are unreachable and dropped.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def icm_equities(stacks: Sequence[float], payouts: Sequence[float]) -> list[float]:
    """Expected prize per player under Malmuth-Harville.

    Recursion: the top remaining prize is awarded to each still-standing player
    with probability stack_p / (sum of standing stacks); conditioning on that
    winner, the rest of the ladder is awarded to the remaining players. Memoised
    on the frozenset of standing players so cost is O(2^N · N), fine for N ≤ 9.
    """
    stacks = [float(s) for s in stacks]
    n = len(stacks)
    # A negative stack is never a legal tournament state; it means a caller
    # subtracted a blind/ante a seat could not actually pay. Left unguarded the
    # recursion happily returns negative $ equity (round-2 finding [E36]), which
    # is nonsense that then propagates silently into an answer key.
    #
    # `s >= 0.0` alone does NOT express that intent (round-3 finding [C2]):
    # `inf >= 0.0` is True, and an infinite stack makes `stacks[p] / total` a
    # NaN that spreads through the whole equity vector without ever tripping an
    # assert. NaN fails the comparison and is rejected here too.
    assert all(math.isfinite(s) and s >= 0.0 for s in stacks), (
        f"stacks must be finite and non-negative, got {stacks}")
    # Prizes are money: negative or non-finite entries are not a ladder, and an
    # empty/zero ladder makes every ICM delta 0 (round-2 finding [E34]).
    assert all(math.isfinite(p) and p >= 0.0 for p in payouts), (
        f"payouts must be finite and non-negative, got {list(payouts)}")
    # Can't pay more places than there are players.
    payouts = [float(p) for p in payouts[:n]]

    memo: dict[frozenset[int], list[float]] = {}

    def solve(active: frozenset[int]) -> list[float]:
        cached = memo.get(active)
        if cached is not None:
            return cached
        place = n - len(active)  # 0-indexed prize place being awarded now
        eq = [0.0] * n
        if not active or place >= len(payouts):
            memo[active] = eq
            return eq
        total = sum(stacks[p] for p in active)
        top = payouts[place]
        if total <= 0.0:
            # Degenerate all-zero stacks: split remaining prizes evenly.
            remaining = payouts[place:]
            share = sum(remaining) / len(active)
            for p in active:
                eq[p] = share
            memo[active] = eq
            return eq
        for p in active:
            frac = stacks[p] / total
            eq[p] += frac * top
            sub = solve(active - {p})
            for q in range(n):
                eq[q] += frac * sub[q]
        memo[active] = eq
        return eq

    return solve(frozenset(range(n)))
