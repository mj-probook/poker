"""Slice G — depth-limited CFR with ORACLE leaf values (impl doc §3 Slice G).

The sanity harness that de-risks ReBeL before any neural net: solve only the
round-1 trunk of Leduc, replacing the round-2 continuation with leaf values read
off a full-game CFR+ solution (the oracle). If the mechanism is correct, the
depth-limited solve reproduces the full game's root VALUE and solves the trunk to
equilibrium. (Turning that into a non-exploitable full agent additionally needs
SAFE re-solving — see test_rebel_safe_resolve.)

Leaf value = value of continuing under the full-game equilibrium σ* (a fixed
blueprint continuation), computed exactly over the public-card deal.
"""

from pokerlab.cfr.cfr import CFRSolver
from pokerlab.cfr.exploit import nash_conv, on_policy_values
from pokerlab.cfr.game import build_tree
from pokerlab.cfr.leduc import CALL, RAISE, LeducPoker
from pokerlab.rebel.depth_limited import (
    DepthLimitedLeducOracle,
    continuation_value,
    round2_entry_state,
)

TRUNK_ITERS = 800


def test_round2_entry_state_is_a_public_deal_node():
    # Constructed leaf: round 2, no public card yet, contributions equal.
    s = round2_entry_state(contrib=(3, 3), bets=(CALL, RAISE, CALL), c0=4, c1=2)
    g = LeducPoker()
    assert s.rnd == 2 and s.public is None and not s.done
    assert g.current_player(s) < 0  # a CHANCE node (public about to be dealt)


def test_continuation_value_is_zero_sum_and_bounded(leduc_sigma_star):
    _, sigma = leduc_sigma_star
    g = LeducPoker()
    s = round2_entry_state(contrib=(3, 3), bets=(CALL, CALL), c0=4, c1=0)
    v0 = continuation_value(g, sigma, s, player=0)
    v1 = continuation_value(g, sigma, s, player=1)
    assert abs(v0 + v1) < 1e-9           # zero-sum continuation
    assert -13.0 <= v0 <= 13.0


def test_depth_limited_tree_truncates_at_round_two(leduc_sigma_star):
    _, sigma = leduc_sigma_star
    tree = build_tree(DepthLimitedLeducOracle(sigma))
    # No node in the trunk ever exposes a public card: round 2 is folded into
    # pseudo-terminals. Every infoset is a round-1 (public=None) infoset.
    assert all("BNone|" in k for k in tree.infoset_actions)
    # Trunk is far smaller than the full 5520-terminal game.
    assert len(tree.nodes) < 600


def test_depth_limited_oracle_reproduces_full_game_value(leduc_sigma_star):
    full_tree, sigma = leduc_sigma_star
    v_full = on_policy_values(full_tree, sigma)[0]

    dl_tree = build_tree(DepthLimitedLeducOracle(sigma))
    s = CFRSolver(dl_tree, plus=True)
    s.run(TRUNK_ITERS)
    v_dl = on_policy_values(dl_tree, s.average_profile())[0]

    assert abs(v_dl - v_full) <= 2e-3


def test_depth_limited_trunk_is_solved_to_equilibrium(leduc_sigma_star):
    # The depth-limited game is a well-defined 2p zero-sum game; trunk CFR must
    # drive ITS OWN NashConv to the M1 bar.
    _, sigma = leduc_sigma_star
    dl_tree = build_tree(DepthLimitedLeducOracle(sigma))
    s = CFRSolver(dl_tree, plus=True)
    s.run(TRUNK_ITERS)
    assert nash_conv(dl_tree, s.average_profile()) <= 1e-3
