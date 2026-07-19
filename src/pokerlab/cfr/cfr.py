"""CFR family (plan §7 L1; impl doc §3 Slice B): vanilla CFR, CFR+.

Runs over the flat `Tree` cache from game.py so each iteration is pure
arithmetic — no game logic re-evaluated. Plain-Python floats in the hot loop
(numpy's per-call overhead dominates on the 2–3-action infosets here).

  vanilla CFR : alternating updates, regret matching, uniform-averaged mean.
  CFR+        : alternating updates, regret-matching-plus (cumulative regrets
                floored at 0 *at infoset level, after* each player's traversal),
                linear (weight-t) strategy averaging.

Correctness note (cost a real bug once): the strategy σ^t is *snapshotted* per
traversal and held fixed, because one infoset spans many tree nodes (different
opponent cards). Regrets accumulate additively during the walk; the RM+ floor is
a separate infoset-level pass afterwards — it cannot be applied per-history
(OpenSpiel's cfr.py makes the same point).

`average_profile()` returns the mean strategy — the object whose nash_conv is the
convergence measure. `current_profile()` exposes the (unaveraged) live strategy.
"""

from __future__ import annotations

from .game import Game, Profile, Tree, build_tree
from .regret import regret_match


class CFRSolver:
    def __init__(self, game: Game | Tree, *, plus: bool = False,
                 alternating: bool = True,
                 linear_averaging: bool | None = None):
        self.tree = game if isinstance(game, Tree) else build_tree(game)
        self.plus = plus
        self.alternating = alternating
        self.linear = plus if linear_averaging is None else linear_averaging
        self.n = self.tree.num_players

        keys = list(self.tree.infoset_actions.keys())
        self._id = {k: i for i, k in enumerate(keys)}
        self._keys = keys
        self.regret = [[0.0] * len(self.tree.infoset_actions[k]) for k in keys]
        self.strategy_sum = [[0.0] * len(self.tree.infoset_actions[k]) for k in keys]

        # Per-node fast fields (avoid attribute lookups in the hot recursion).
        nodes = self.tree.nodes
        self._kind = [n.kind for n in nodes]
        self._player = [n.player for n in nodes]
        self._children = [n.children for n in nodes]
        self._probs = [n.chance_probs for n in nodes]
        self._payoff = [n.payoff for n in nodes]
        self._iset = [self._id[n.infoset] if n.kind == "decision" else -1
                      for n in nodes]
        self.t = 0

    # -- regret matching ----------------------------------------------------- #
    def _strategy(self, iset: int) -> list[float]:
        return regret_match(self.regret[iset])

    def _snapshot(self) -> list[list[float]]:
        return [self._strategy(i) for i in range(len(self._keys))]

    def _rm_plus_reset(self) -> None:
        for reg in self.regret:
            for ai in range(len(reg)):
                if reg[ai] < 0.0:
                    reg[ai] = 0.0

    # -- one CFR traversal (σ fixed via `strat` snapshot) -------------------- #
    def _traverse(self, idx: int, reach: list[float], up: int,
                  strat: list[list[float]]) -> list[float]:
        kind = self._kind[idx]
        if kind == "terminal":
            return list(self._payoff[idx])
        children = self._children[idx]
        if kind == "chance":
            probs = self._probs[idx]
            v = [0.0] * self.n
            for child, p in zip(children, probs):
                cv = self._traverse(child, reach[:-1] + [reach[-1] * p], up, strat)
                for i in range(self.n):
                    v[i] += p * cv[i]
            return v

        p = self._player[idx]
        iset = self._iset[idx]
        sigma = strat[iset]
        v = [0.0] * self.n
        child_p = [0.0] * len(children)
        for ai, child in enumerate(children):
            nr = reach[:]
            nr[p] *= sigma[ai]
            cv = self._traverse(child, nr, up, strat)
            s = sigma[ai]
            for i in range(self.n):
                v[i] += s * cv[i]
            child_p[ai] = cv[p]

        if p == up or up < 0:  # up<0 => simultaneous (update every player)
            cf = reach[-1]
            for q in range(self.n):
                if q != p:
                    cf *= reach[q]
            reg = self.regret[iset]
            for ai in range(len(reg)):
                reg[ai] += cf * (child_p[ai] - v[p])
            w = (self.t if self.linear else 1.0) * reach[p]
            if w != 0.0:
                ss = self.strategy_sum[iset]
                for ai in range(len(ss)):
                    ss[ai] += w * sigma[ai]
        return v

    def iterate(self) -> None:
        self.t += 1
        base = [1.0] * self.n + [1.0]  # per-player reach + chance reach
        if self.alternating:
            for up in range(self.n):
                strat = self._snapshot()  # σ^t fixed for this player's walk
                for root, prob in self.tree.roots:
                    r = base[:]
                    r[-1] = prob
                    self._traverse(root, r, up, strat)
                if self.plus:
                    self._rm_plus_reset()
        else:
            strat = self._snapshot()
            for root, prob in self.tree.roots:
                r = base[:]
                r[-1] = prob
                self._traverse(root, r, -1, strat)
            if self.plus:
                self._rm_plus_reset()

    def run(self, iters: int) -> None:
        for _ in range(iters):
            self.iterate()

    # -- strategies ---------------------------------------------------------- #
    def _profile_from(self, tables: list[list[float]]) -> Profile:
        prof: Profile = {}
        for i, key in enumerate(self._keys):
            actions = self.tree.infoset_actions[key]
            row = tables[i]
            s = sum(row)
            if s > 0.0:
                prof[key] = {a: row[ai] / s for ai, a in enumerate(actions)}
            else:
                u = 1.0 / len(actions)
                prof[key] = {a: u for a in actions}
        return prof

    def average_profile(self) -> Profile:
        return self._profile_from(self.strategy_sum)

    def current_profile(self) -> Profile:
        return self._profile_from(self._snapshot())
