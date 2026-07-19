"""External-sampling MCCFR (plan §7 L1; impl doc §3 Slice B).

Canonical external sampling (Lanctot et al. 2009): for each traverser i, sample
chance and opponent actions, recurse over ALL of i's own actions. Regrets update
at i's nodes (unweighted — sampling supplies the reach); the average strategy
accumulates at opponent nodes. Seeded RNG => deterministic tests.

The ≤5e-3 exploitability bar is impl doc §3 Slice B's, NOT a milestone exit:
PLAN pins only the 1e-3 CFR+ bar (M1). Sampling converges slower than the full
tree walk, so MCCFR is held to the looser Slice-B figure.
"""

from __future__ import annotations

import random

from .game import Game, Profile, Tree, build_tree
from .regret import regret_match


class MCCFRSolver:
    def __init__(self, game: Game | Tree, *, seed: int = 0):
        self.tree = game if isinstance(game, Tree) else build_tree(game)
        self.n = self.tree.num_players
        keys = list(self.tree.infoset_actions.keys())
        self._id = {k: i for i, k in enumerate(keys)}
        self._keys = keys
        self.regret = [[0.0] * len(self.tree.infoset_actions[k]) for k in keys]
        self.strategy_sum = [[0.0] * len(self.tree.infoset_actions[k]) for k in keys]

        nodes = self.tree.nodes
        self._kind = [n.kind for n in nodes]
        self._player = [n.player for n in nodes]
        self._children = [n.children for n in nodes]
        self._probs = [n.chance_probs for n in nodes]
        self._payoff = [n.payoff for n in nodes]
        self._iset = [self._id[n.infoset] if n.kind == "decision" else -1
                      for n in nodes]
        self.rng = random.Random(seed)
        self.t = 0

    def _strategy(self, iset: int) -> list[float]:
        return regret_match(self.regret[iset])

    def _sample(self, probs: list[float]) -> int:
        x = self.rng.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if x < acc:
                return i
        return len(probs) - 1

    def _walk(self, idx: int, traverser: int) -> float:
        kind = self._kind[idx]
        if kind == "terminal":
            return self._payoff[idx][traverser]
        children = self._children[idx]
        if kind == "chance":
            probs = self._probs[idx]
            return self._walk(children[self._sample(probs)], traverser)

        iset = self._iset[idx]
        sigma = self._strategy(iset)
        if self._player[idx] == traverser:
            child_util = [self._walk(c, traverser) for c in children]
            node_util = sum(s * u for s, u in zip(sigma, child_util))
            reg = self.regret[iset]
            for ai in range(len(reg)):
                reg[ai] += child_util[ai] - node_util
            return node_util
        # opponent: accumulate average strategy, sample one action
        ss = self.strategy_sum[iset]
        for ai in range(len(ss)):
            ss[ai] += sigma[ai]
        return self._walk(children[self._sample(sigma)], traverser)

    def iterate(self) -> None:
        self.t += 1
        for traverser in range(self.n):
            for root, _prob in self.tree.roots:
                self._walk(root, traverser)

    def run(self, iters: int) -> None:
        for _ in range(iters):
            self.iterate()

    def average_profile(self) -> Profile:
        prof: Profile = {}
        for i, key in enumerate(self._keys):
            actions = self.tree.infoset_actions[key]
            row = self.strategy_sum[i]
            s = sum(row)
            if s > 0.0:
                prof[key] = {a: row[ai] / s for ai, a in enumerate(actions)}
            else:
                u = 1.0 / len(actions)
                prof[key] = {a: u for a in actions}
        return prof
