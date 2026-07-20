"""Slice B — Leduc hold'em game-tree correctness + CFR+ convergence (impl doc §3).

Hand-derived payoff/legal-action fixtures pin the game; then CFR+ must reach the
M1 exit bar (NashConv ≤ 1e-3). Structural counts (5520 leaves / 936 infosets /
±13 range) are the same partition OpenSpiel uses; the openspiel marker replays
every terminal to prove unit parity.
"""

from pokerlab.cfr.cfr import CFRSolver
from pokerlab.cfr.exploit import nash_conv
from pokerlab.cfr.game import build_tree
from pokerlab.cfr.leduc import CALL, FOLD, RAISE, LeducPoker, LeducState


def test_leduc_structural_counts_and_units():
    tree = build_tree(LeducPoker())
    terms = [n for n in tree.nodes if n.kind == "terminal"]
    assert len(terms) == 5520
    assert len(tree.infoset_actions) == 936
    p0 = sorted({n.payoff[0] for n in terms})
    assert p0[0] == -13.0 and p0[-1] == 13.0
    assert all(sum(n.payoff) == 0.0 for n in terms)


def test_leduc_legal_actions_follow_betting_rules():
    g = LeducPoker()
    # round start, contributions equal -> Call / Raise (no Fold).
    s0 = LeducState(private=(4, 2), contrib=(1, 1), raises=0)
    assert g.legal_actions(s0) == [CALL, RAISE]
    # facing a bet, below cap -> Fold / Call / Raise.
    s1 = LeducState(private=(4, 2), contrib=(3, 1), raises=1)
    assert g.legal_actions(s1) == [FOLD, CALL, RAISE]
    # raise cap reached -> Fold / Call only.
    s2 = LeducState(private=(4, 2), contrib=(5, 3), raises=2)
    assert g.legal_actions(s2) == [FOLD, CALL]


def test_leduc_hand_derived_showdown_payoffs():
    g = LeducPoker()
    # higher rank, neither pairs: K vs Q on J board, checked down -> +1 to P0.
    assert g.returns(LeducState(private=(4, 2), public=0, contrib=(1, 1))) == [1.0, -1.0]
    # pairing beats a higher card: P0 J pairs J-board vs P1 K -> +3 to P0.
    assert g.returns(LeducState(private=(0, 4), public=1, contrib=(3, 3))) == [3.0, -3.0]
    # equal ranks, neither pairs -> split.
    assert g.returns(LeducState(private=(2, 3), public=4, contrib=(5, 5))) == [0.0, 0.0]


def test_leduc_hand_derived_fold_payoff():
    g = LeducPoker()
    # P0 raised (contrib 3), P1 folded (contrib 1): P0 wins P1's 1 -> +1.
    s = LeducState(private=(0, 4), public=None, contrib=(3, 1), folder=1, done=True)
    assert g.is_terminal(s)
    assert g.returns(s) == [1.0, -1.0]


def test_cfr_plus_leduc_reaches_1e_3():
    tree = build_tree(LeducPoker())
    s = CFRSolver(tree, plus=True)
    s.run(1000)  # crosses 1e-3 near ~650 iters; 1000 -> ~4.7e-4 in ~6s (<<30s)
    assert nash_conv(tree, s.average_profile()) <= 1e-3
