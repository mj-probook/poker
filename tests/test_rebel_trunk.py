"""Slice G — belief-vectorized depth-limited CFR, the ReBeL trunk (impl doc §3).

The trunk solves round 1 over *ranges* (vectors over the 6 private cards), and at
each round-2 leaf queries a leaf-value function with the current belief.
`fixed_continuation_leaf_values` supplies belief-LINEAR leaves read off the
full-game solution σ*; the vectorized engine must then solve the SAME
depth-limited game as the per-deal `DepthLimitedLeducOracle` reference (a check
against the tested `CFRSolver`) and reproduce the full game's root value.
"""

from pokerlab.cfr.cfr import CFRSolver
from pokerlab.cfr.exploit import nash_conv, on_policy_values
from pokerlab.cfr.game import build_tree
from pokerlab.rebel.depth_limited import DepthLimitedLeducOracle
from pokerlab.rebel.trunk import DepthLimitedSolver, fixed_continuation_leaf_values


def test_vectorized_trunk_matches_per_deal_oracle_under_fixed_leaves(leduc_sigma_star):
    # Belief-linear (fixed σ*) leaves: the vectorized engine must solve the SAME
    # depth-limited game as the per-deal reference, to the same value/equilibrium.
    _, sigma = leduc_sigma_star
    dl_tree = build_tree(DepthLimitedLeducOracle(sigma))
    ref = CFRSolver(dl_tree, plus=True)
    ref.run(800)
    v_ref = on_policy_values(dl_tree, ref.average_profile())[0]

    solver = DepthLimitedSolver(fixed_continuation_leaf_values(sigma))
    solver.run(800)
    sigma1 = solver.round1_profile()

    v_vec = on_policy_values(dl_tree, sigma1)[0]
    assert abs(v_vec - v_ref) < 2e-3            # same game value
    assert nash_conv(dl_tree, sigma1) <= 1e-3   # engine solved it to equilibrium


def test_vectorized_trunk_reproduces_full_game_root_value(leduc_sigma_star):
    # With oracle (σ* continuation) leaves the depth-limited trunk's root value
    # matches the full game — the value-side of the oracle-leaf sanity harness.
    full_tree, sigma = leduc_sigma_star
    v_full = on_policy_values(full_tree, sigma)[0]

    solver = DepthLimitedSolver(fixed_continuation_leaf_values(sigma))
    solver.run(800)
    dl_tree = build_tree(DepthLimitedLeducOracle(sigma))
    v_vec = on_policy_values(dl_tree, solver.round1_profile())[0]
    assert abs(v_vec - v_full) <= 2e-3
