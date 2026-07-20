"""§6.1 parameterized perturbation bot (plan §6.1; impl doc §3 Slice H).

Knobs (VPIP/aggression/bluff) over a base `Solution`/strategy table produce a
*queryable* strategy that deviates from equilibrium — the fixed opponent the M6
exact best-response exploiter attacks (plan §6.3). Because the bot's strategy is
an explicit distribution, an exact tabular BR is computable directly and its
"measured exploitability" is exact.

Two faces:
  * `profile(infoset_actions)` -> a normalized `Profile` over a toy game, fed to
    `cfr.best_response` for the exact-BR exit.
  * `act(hand, rng)` -> a legal engine `Action`, so the same bot can be a
    `VecEnv` villain (no base table needed; label-based aggression over a
    uniform base).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from pokerlab.engine.state import Hand
from pokerlab.env.encoding import legal_engine_actions
from pokerlab.types import Action

AGGRESSIVE_LABELS = frozenset({"bet", "raise", "allin"})
FOLD_LABEL = "fold"

Dist = dict[Any, float]


def perturb_dist(
    dist: Dist,
    aggressive: set,
    *,
    aggression: float = 0.0,
    loosen: float = 0.0,
    fold: Any | None = None,
) -> Dist:
    """Shift a base action distribution by the knobs, staying a valid pmf.

    ``loosen`` (VPIP, 0..1) moves that fraction of fold mass onto non-fold
    actions. ``aggression`` (-1..1) moves that fraction between passive and
    aggressive actions (bluff = aggression applied to the base). Mass is
    reallocated proportionally, then renormalized.
    """
    out = dict(dist)
    if fold is not None and fold in out and loosen > 0 and len(out) > 1:
        moved = out[fold] * loosen
        out[fold] -= moved
        others = [a for a in out if a != fold]
        om = sum(out[a] for a in others)
        for a in others:
            out[a] += moved * (out[a] / om if om > 0 else 1.0 / len(others))
    agg = [a for a in out if a in aggressive]
    pas = [a for a in out if a not in aggressive and a != fold]
    if agg and pas and aggression > 0:
        moved = sum(out[a] for a in pas) * aggression
        for a in pas:
            out[a] *= 1.0 - aggression
        am = sum(out[a] for a in agg)
        for a in agg:
            out[a] += moved * (out[a] / am if am > 0 else 1.0 / len(agg))
    elif agg and pas and aggression < 0:
        k = -aggression
        moved = sum(out[a] for a in agg) * k
        for a in agg:
            out[a] *= 1.0 - k
        pm = sum(out[a] for a in pas)
        for a in pas:
            out[a] += moved * (out[a] / pm if pm > 0 else 1.0 / len(pas))
    total = sum(out.values()) or 1.0
    return {a: out[a] / total for a in out}


class PerturbationBot:
    def __init__(
        self,
        base: dict[str, Dist] | None = None,
        *,
        aggression: float = 0.0,
        loosen: float = 0.0,
        aggressive: set | Callable[[Any], bool] = AGGRESSIVE_LABELS,
        fold: Any | None = FOLD_LABEL,
    ) -> None:
        self.base = base or {}
        self.aggression = aggression
        self.loosen = loosen
        self.aggressive = aggressive
        self.fold = fold

    def _agg_set(self, actions: Iterable) -> set:
        if callable(self.aggressive):
            return {a for a in actions if self.aggressive(a)}
        return {a for a in actions if a in self.aggressive}

    def _dist(self, actions: list, base: Dist | None) -> Dist:
        if base is None:
            base = {a: 1.0 / len(actions) for a in actions}  # uniform fallback
        return perturb_dist(
            base, self._agg_set(actions),
            aggression=self.aggression, loosen=self.loosen, fold=self.fold,
        )

    def profile(self, infoset_actions: dict[str, list]) -> dict[str, Dist]:
        """Perturbed strategy over every infoset (a `cfr` Profile)."""
        return {
            key: self._dist(actions, self.base.get(key))
            for key, actions in infoset_actions.items()
        }

    def act(self, hand: Hand, rng) -> Action:
        """Sample a legal engine action (VecEnv villain path)."""
        legal = [a for _, a in legal_engine_actions(hand)]
        dist = self._dist(legal, None)
        r = rng.random()
        acc = 0.0
        for a in legal:
            acc += dist[a]
            if r <= acc:
                return a
        return legal[-1]
