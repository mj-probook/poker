"""Slice D — turn+river subgame (one river chance layer) (impl §3 Slice D).

Adds a chance layer on top of the verified river machinery. The new risk is
chance-node card removal + the runout normalization, so the no-betting turn
game is pinned against an independently computed equity, and the betting solver
is shown to drive exploitability down.
"""

from __future__ import annotations

import numpy as np

from pokerlab.engine.cards import card_from_str
from pokerlab.engine.evaluator import rank_showdown
from pokerlab.solver import subgame as sg


def _cards(s: str) -> tuple[int, ...]:
    return tuple(card_from_str(s[i:i + 2]) for i in range(0, len(s), 2))


def _range(combos: list[str]) -> np.ndarray:
    r = np.zeros(sg.NUM_COMBOS)
    for cs in combos:
        a, b = _cards(cs)
        r[sg.COMBO_INDEX[(min(a, b), max(a, b))]] = 1.0
    return r


TURN = _cards("AsKd7h2c")  # 4-card board
R0 = _range(["QhQs", "JdJc", "TdTh", "5c5d"])
R1 = _range(["8d8h", "6s6d", "AcAh", "Kh9d"])


def _independent_equity(board, r0, r1, pot0):
    supp0 = [i for i in range(sg.NUM_COMBOS) if r0[i] > 0]
    supp1 = [i for i in range(sg.NUM_COMBOS) if r1[i] > 0]
    num = den = 0.0
    for h0 in supp0:
        c0 = set(sg.COMBOS[h0])
        for h1 in supp1:
            c1 = set(sg.COMBOS[h1])
            if c0 & c1 or (c0 | c1) & set(board):
                continue
            rivers = [c for c in range(52) if c not in board and c not in c0 and c not in c1]
            acc = 0.0
            for rc in rivers:
                fb = board + (rc,)
                s0 = rank_showdown(sg.COMBOS[h0], fb)
                s1 = rank_showdown(sg.COMBOS[h1], fb)
                acc += 1.0 if s0 < s1 else (0.5 if s0 == s1 else 0.0)
            w = r0[h0] * r1[h1]
            num += w * acc / len(rivers)
            den += w
    return pot0 * num / den


def test_no_betting_turn_game_equals_true_equity():
    # OOP can only check; both check down -> OOP wins its pot-share equity.
    cfg = sg.BetConfig(sizes=(), jam=False, max_raises=0)
    tree = sg.build_tree(TURN, pot0=8.0, stack=20.0, cfg=cfg)
    s = sg.SubgameSolver(tree, TURN, R0.copy(), R1.copy(), pot0=8.0)
    s.iterate(1)  # single pass; no real decisions to train
    ev0 = s.on_policy_value(0)
    expected = _independent_equity(TURN, s.range0, s.range1, pot0=8.0)
    # `ev0` is OOP's share of the pot in POT UNITS (bb), so it must land strictly
    # inside (0, pot0) — a share, never the whole pot and never nothing. This is
    # deliberately independent of `_independent_equity` below: the equality check
    # would pass if the solver and the reference helper were rescaled the same
    # wrong way, which is exactly how the round-1 [F1] units bug hid.
    # (Replaces `assert ev0 == 0 or True`, which was true for every possible ev0.)
    assert 0.0 < ev0 < 8.0
    assert abs(ev0 - expected) < 1e-6


def test_turn_solver_reduces_exploitability():
    cfg = sg.BetConfig(sizes=(0.75,), jam=False, max_raises=0)
    tree = sg.build_tree(TURN, pot0=8.0, stack=20.0, cfg=cfg)
    s = sg.SubgameSolver(tree, TURN, R0.copy(), R1.copy(), pot0=8.0)
    s.iterate(2)
    early = s.exploitability_pct()
    s.iterate(800)
    late = s.exploitability_pct()
    assert late < early
    assert late < 0.005  # ≤0.5% of pot
