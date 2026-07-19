"""Slice B — Kuhn poker game-tree correctness (impl doc §3).

Structural + hand-derived-payoff fixtures. No solver here; this pins the game
before anything (instrument or CFR) is trusted on it.
"""

from pokerlab.cfr.game import build_tree
from pokerlab.cfr.kuhn import KuhnPoker, KuhnState


def _terminals(tree):
    return [n for n in tree.nodes if n.kind == "terminal"]


def test_kuhn_tree_leaf_and_infoset_counts():
    tree = build_tree(KuhnPoker())
    # 6 deals (3x2) x 5 terminal histories = 30 leaves (cf. OpenSpiel).
    assert len(_terminals(tree)) == 30
    # 12 infosets: 3 cards x {'', 'pb'} for P0, 3 cards x {'p', 'b'} for P1.
    assert len(tree.infoset_actions) == 12


def test_kuhn_infoset_key_format_matches_openspiel():
    g = KuhnPoker()
    # P0 dealt K(2), P1 dealt Q(1): after pass,bet P0's infoset is "2pb".
    s = KuhnState(cards=(2, 1), history=(0, 1))
    assert g.current_player(s) == 0
    assert g.infoset_key(s, 0) == "2pb"
    # P1 facing an opening bet with J(0): "0b".
    s2 = KuhnState(cards=(2, 0), history=(1,))
    assert g.current_player(s2) == 1
    assert g.infoset_key(s2, 1) == "0b"


def test_kuhn_hand_derived_payoffs():
    g = KuhnPoker()
    # showdown pass-pass, K beats Q -> +1 to P0.
    assert g.returns(KuhnState((2, 1), (0, 0))) == [1.0, -1.0]
    # bet, fold -> P0 wins the ante, +1 regardless of cards.
    assert g.returns(KuhnState((0, 2), (1, 0))) == [1.0, -1.0]
    # bet, call showdown for 2, J loses to K -> -2 to P0.
    assert g.returns(KuhnState((0, 2), (1, 1))) == [-2.0, 2.0]
    # pass, bet, fold -> P0 folds, -1.
    assert g.returns(KuhnState((2, 0), (0, 1, 0))) == [-1.0, 1.0]
    # pass, bet, call showdown for 2, K beats J -> +2.
    assert g.returns(KuhnState((2, 0), (0, 1, 1))) == [2.0, -2.0]


def test_kuhn_zero_sum_over_all_terminals():
    tree = build_tree(KuhnPoker())
    for leaf in _terminals(tree):
        assert sum(leaf.payoff) == 0.0
