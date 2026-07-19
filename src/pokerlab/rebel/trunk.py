"""Belief-vectorized depth-limited CFR — the ReBeL trunk (plan §7 L3; §3 Slice G).

This is the ReBeL mechanism proper. The trunk is Leduc round 1 solved over
*ranges*: each infoset's regrets/strategy are vectors over the 6 private cards,
and CFR traverses the small public betting tree once per iteration carrying both
players' reach vectors (r0, r1). At a round-2 leaf it calls a `leaf_value_fn`
with the current belief and gets back per-card counterfactual values.

The leaf value MUST be belief-conditioned (the equilibrium value at the current
range) for the resulting agent to be non-exploitable — a fixed blueprint
continuation misprices the leaf whenever the opponent would deviate in round 2
(see tests/test_rebel_trunk.py). `fixed_continuation_leaf_values` (belief-linear,
blueprint σ*) exists only as a cross-check against the per-deal reference; the
real oracle and the value net are belief-conditioned.

Counterfactual-value convention (shared with resolve.py): cfv_p[c] =
Σ_{c'} DEAL[c,c'] · r_opp[c'] · U_p(c,c'), own reach excluded, DEAL = 1/30 prior.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pokerlab.cfr.game import Profile
from pokerlab.cfr.leduc import LeducPoker, LeducState
from pokerlab.cfr.regret import regret_match_np

from .depth_limited import continuation_value, round2_entry_state
from .pbs import DEAL, NUM_CARDS

_BASE = LeducPoker()


@dataclass
class _Node:
    kind: str                       # "decision" | "fold" | "leaf"
    player: int = -1
    infoset: tuple | None = None    # (player, bets)
    contrib: tuple = (1, 1)
    bets: tuple = ()
    actions: list = field(default_factory=list)
    children: list = field(default_factory=list)
    fold_p0: float = 0.0


def build_round1_tree() -> tuple[list[_Node], int]:
    """Public round-1 betting tree (card-independent): decisions, folds, leaves."""
    nodes: list[_Node] = []

    def build(s: LeducState) -> int:
        idx = len(nodes)
        if s.done:                                  # fold terminal
            amt = float(s.contrib[s.folder])
            p0 = amt if (1 - s.folder) == 0 else -amt
            nodes.append(_Node("fold", fold_p0=p0))
            return idx
        if s.rnd == 2 and s.public is None:         # round-2 entry leaf
            nodes.append(_Node("leaf", contrib=s.contrib, bets=s.bets))
            return idx
        actions = _BASE.legal_actions(s)
        node = _Node("decision", player=s.to_move, infoset=(s.to_move, s.bets),
                     contrib=s.contrib, bets=s.bets, actions=actions)
        nodes.append(node)
        node.children = [build(_BASE.apply(s, a)) for a in actions]
        return idx

    root = build(LeducState(private=(0, 1)))  # dummy cards; betting ignores them
    return nodes, root


class DepthLimitedSolver:
    """CFR+ over the round-1 range tree; leaf values from `leaf_value_fn`."""

    def __init__(self, leaf_value_fn, *, linear: bool = True):
        self.nodes, self.root = build_round1_tree()
        self.leaf_value_fn = leaf_value_fn
        self.linear = linear
        self.regret: dict[tuple, np.ndarray] = {}
        self.strat_sum: dict[tuple, np.ndarray] = {}
        self.actions_for: dict[tuple, list] = {}
        for n in self.nodes:
            if n.kind == "decision" and n.infoset not in self.regret:
                a = len(n.actions)
                self.regret[n.infoset] = np.zeros((NUM_CARDS, a))
                self.strat_sum[n.infoset] = np.zeros((NUM_CARDS, a))
                self.actions_for[n.infoset] = n.actions
        self.t = 0
        self._leaf_cache: dict = {}
        self.W = DEAL.copy()  # 1/30 off-diagonal, 0 on diagonal (blocking)

    def _strategy(self, iset: tuple) -> np.ndarray:
        # regret is laid out (card, action), so actions are axis 1
        return regret_match_np(self.regret[iset], axis=1)

    def _snapshot(self) -> dict[tuple, np.ndarray]:
        return {iset: self._strategy(iset) for iset in self.regret}

    def _traverse(self, idx, r0, r1, snap: dict, up: int):
        """σ fixed via `snap`; accumulate regret/strategy only for player `up`."""
        n = self.nodes[idx]
        if n.kind == "fold":
            return n.fold_p0 * (self.W @ r1), -n.fold_p0 * (self.W @ r0)
        if n.kind == "leaf":
            # Both alternating passes of an iterate share the snapshot belief, so
            # the same leaf is queried with identical reaches — memoize per iterate
            # (halves value-oracle / net calls, which dominate cost).
            ck = (n.bets, r0.tobytes(), r1.tobytes())
            cfv = self._leaf_cache.get(ck)
            if cfv is None:
                cfv = self.leaf_value_fn(n.contrib, n.bets, r0, r1)
                self._leaf_cache[ck] = cfv
            return cfv[0], cfv[1]

        iset, p = n.infoset, n.player
        sigma = snap[iset]
        child0, child1 = [], []
        for ai in range(len(n.actions)):
            if p == 0:
                c0, c1 = self._traverse(n.children[ai], r0 * sigma[:, ai], r1, snap, up)
            else:
                c0, c1 = self._traverse(n.children[ai], r0, r1 * sigma[:, ai], snap, up)
            child0.append(c0)
            child1.append(c1)
        m0 = np.stack(child0, axis=1)  # (6, A)  P0 cfv per action
        m1 = np.stack(child1, axis=1)

        if p == 0:
            cfv_p, cfv_q, own, mp = (m0 * sigma).sum(1), m1.sum(1), r0, m0
            out = (cfv_p, cfv_q)
        else:
            cfv_p, cfv_q, own, mp = (m1 * sigma).sum(1), m0.sum(1), r1, m1
            out = (cfv_q, cfv_p)          # always (cfv0, cfv1)
        if p == up:
            self.regret[iset] += mp - cfv_p[:, None]
            w = (self.t if self.linear else 1.0) * own
            self.strat_sum[iset] += w[:, None] * sigma
        return out

    def iterate(self) -> None:
        self.t += 1
        self._leaf_cache = {}
        ones = np.ones(NUM_CARDS)
        for up in (0, 1):                              # alternating updates
            snap = self._snapshot()
            self._traverse(self.root, ones, ones, snap, up)
            for reg in self.regret.values():           # RM+ floor
                np.clip(reg, 0.0, None, out=reg)

    def run(self, iters: int) -> None:
        for _ in range(iters):
            self.iterate()

    def round1_profile(self) -> Profile:
        prof: Profile = {}
        for iset, ss in self.strat_sum.items():
            _, bets = iset
            actions = self.actions_for[iset]
            s = ss.sum(axis=1, keepdims=True)
            avg = np.where(s > 0.0, ss / np.where(s > 0.0, s, 1.0),
                           1.0 / ss.shape[1])
            for c in range(NUM_CARDS):
                key = f"P{c}|BNone|{bets}"
                prof[key] = {actions[ai]: float(avg[c, ai])
                             for ai in range(len(actions))}
        return prof


# --------------------------------------------------------------------------- #
# Leaf-value oracles.
# --------------------------------------------------------------------------- #
def fixed_continuation_leaf_values(sigma_star: Profile):
    """Belief-LINEAR leaf values = the counterfactual values σ* actually realizes.

    cfv_p[c] = Σ_{c'} DEAL·r_opp[c'] · U_p(c,c') with U from σ*'s round-2 play — the
    "leaf values from the full CFR+ solution" the oracle harness uses. These are
    the true equilibrium continuation values (belief-linear in a fixed blueprint),
    which the depth-limited trunk reproduces the root value against and which the
    safe-resolve gadget preserves to reproduce the full equilibrium. The value net
    (valuenet.py) learns to approximate this map.
    """
    cache: dict[tuple, np.ndarray] = {}

    def matrix(contrib: tuple, bets: tuple) -> np.ndarray:
        key = (contrib, bets)
        a0 = cache.get(key)
        if a0 is None:
            u0 = np.zeros((NUM_CARDS, NUM_CARDS))
            for c0 in range(NUM_CARDS):
                for c1 in range(NUM_CARDS):
                    if c0 != c1:
                        u0[c0, c1] = continuation_value(
                            _BASE, sigma_star,
                            round2_entry_state(contrib, bets, c0, c1), 0)
            a0 = DEAL * u0
            cache[key] = a0
        return a0

    def leaf_fn(contrib, bets, r0, r1) -> np.ndarray:
        a0 = matrix(tuple(contrib), tuple(bets))
        return np.stack([a0 @ r1, -(a0.T @ r0)])

    return leaf_fn
