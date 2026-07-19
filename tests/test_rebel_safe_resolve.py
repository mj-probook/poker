"""Slice G — SAFE round-2 re-solving: the oracle-leaf sanity harness (§3).

This is the ReBeL mechanism done correctly. Naive decomposed re-solving is
unsafe (test_rebel_resolve.test_naive_decomposed_resolve_is_unsafe). The CFR-D
gadget preserves the opponent's counterfactual values, and — fed a genuine
equilibrium round-1 plus the leaf values read off the full CFR+ solution (σ*
continuation) — reproduces the full game's exploitability to within the M5 bar.

The critical, hard-won detail: the values the gadget preserves must be the
BLUEPRINT's continuation CFVs (what σ* actually realizes), not a fresh isolated
re-solve's equilibrium value — those differ at low-reach infosets and the
difference is the whole ballgame.
"""

import numpy as np
import pytest

import pytest

from pokerlab.cfr.exploit import nash_conv
from pokerlab.rebel.resolve import leaf_reach, resolve_round2, round2_leaf_lines
from pokerlab.rebel.safe_resolve import safe_rebel_agent_profile, safe_resolve_round2
from pokerlab.rebel.trunk import fixed_continuation_leaf_values


def test_safe_resolve_covers_both_players_round2_infosets(leduc_sigma_star):
    # A single line: the gadget (run per resolver) returns round-2 strategies for
    # both players' infosets on that line, each a valid distribution.
    _, sigma = leduc_sigma_star
    round1 = {k: v for k, v in sigma.items() if "|BNone|" in k}
    ln = round2_leaf_lines()[0]
    r = leaf_reach(round1, ln)
    cfv = resolve_round2(ln["contrib"], ln["bets"], r[0], r[1], iters=300).cfv
    strat = safe_resolve_round2(ln["contrib"], ln["bets"], r[0], r[1], cfv, iters=300)
    assert len(strat) > 0
    for dist in strat.values():
        assert abs(sum(dist.values()) - 1.0) < 1e-6


@pytest.mark.slow
def test_oracle_leaf_safe_resolve_reaches_M5_bar(leduc_sigma_star):
    # THE M5 oracle-leaf harness: leaf values from the full CFR+ solution +
    # SAFE re-solving reproduce full-game exploitability ≤ 2e-3.
    full_tree, sigma = leduc_sigma_star
    round1 = {k: v for k, v in sigma.items() if "|BNone|" in k}
    value_fn = fixed_continuation_leaf_values(sigma)   # leaf values from σ*
    agent = safe_rebel_agent_profile(round1, value_fn, resolve_iters=500)
    assert nash_conv(full_tree, agent) <= 2e-3
