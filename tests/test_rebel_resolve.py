"""Slice G — round-2 subgame re-solving + the oracle ReBeL agent (impl doc §3).

Round 2 is Leduc's last street: its subgames have real terminals, so given a
belief (the range entering round 2) they solve exactly. This is (a) the true
ReBeL value oracle — solve the subgame at a PBS, read off per-card counterfactual
values — and (b) the piece that makes a depth-limited agent non-exploitable:
instead of pasting a blueprint's round-2 play onto a new round-1 line (the
off-belief trap), re-solve round 2 for the belief the trunk actually produced.

The capstone test: trunk (oracle leaves) + re-solved round 2 reproduces the full
game's exploitability to within the M5 bar — with the ORACLE, before any net.
"""

import numpy as np

from pokerlab.cfr.cfr import CFRSolver
from pokerlab.cfr.exploit import nash_conv
from pokerlab.cfr.game import build_tree
from pokerlab.rebel.resolve import (
    LeducRound2Subgame,
    rebel_agent_profile,
    resolve_round2,
    round2_leaf_lines,
)


def test_round2_leaf_lines_enumerates_the_five_betting_closes():
    lines = round2_leaf_lines()
    # Round-1 lines that reach round 2 (no fold): check-check, check-raise-call,
    # check-raise-raise-call, raise-call, raise-raise-call.
    assert len(lines) == 5
    contribs = sorted(ln["contrib"] for ln in lines)
    assert contribs == [(1, 1), (3, 3), (3, 3), (5, 5), (5, 5)]


def test_resolve_solves_the_subgame_to_equilibrium():
    # Uniform ranges entering the smallest pot.
    r = np.ones((2, 6))
    sol = resolve_round2(contrib=(1, 1), bets=(1, 1), reach0=r[0], reach1=r[1],
                         iters=600)
    sub_tree = build_tree(LeducRound2Subgame((1, 1), (1, 1),
                                             sol.belief))
    assert nash_conv(sub_tree, sol.strategy) <= 1e-3


def test_resolve_counterfactual_values_are_zero_sum_at_uniform_belief():
    r = np.ones((2, 6))
    sol = resolve_round2(contrib=(3, 3), bets=(2, 1), reach0=r[0], reach1=r[1],
                         iters=600)
    # With full reach, summed per-card cfv is each player's root value; zero-sum.
    assert abs(sol.cfv[0].sum() + sol.cfv[1].sum()) < 5e-3
    assert sol.cfv.shape == (2, 6)


def test_naive_decomposed_resolve_is_unsafe(leduc_sigma_star):
    # DISCOVERY / regression guard: re-solving round 2 for the belief WITHOUT
    # preserving counterfactual values is unsafe — even σ*'s own genuine round-1
    # becomes exploitable, because low-reach round-2 play is arbitrary and a
    # full-game best-responder deviates in round 1 to punish it. The safe fix
    # (CFR-D gadget) lives in tests/test_rebel_safe_resolve.py.
    full_tree, sigma = leduc_sigma_star
    round1 = {k: v for k, v in sigma.items() if "|BNone|" in k}
    agent = rebel_agent_profile(round1, resolve_iters=400)
    assert nash_conv(full_tree, agent) > 0.05     # provably unsafe
