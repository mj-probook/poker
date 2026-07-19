"""Regret matching — the one place the CFR strategy rule lives (wave-2 [Q4]).

Every CFR variant in this repo turns a regret vector into a strategy the same
way: clip to positives, normalize; if nothing is positive, play uniform. That
rule was written out four times (vanilla CFR, MCCFR, the M3 subgame solver, the
ReBeL trunk), so a change to the convention had four places to miss.

Two entry points rather than one, and the split is measured, not stylistic:

  * `regret_match` — pure-Python, for the tabular solvers whose regret vectors
    are short lists in the innermost recursion.
  * `regret_match_np` — vectorized, for the array-shaped solvers.

Routing the tabular solvers through numpy would unify the signature at 6.8x
per call (0.22us -> 1.47us measured on a 3-action vector); vanilla CFR+ on
Leduc is already the slowest test in the fast suite, so that cost is not
affordable. Both functions implement the identical rule and are tested against
each other, which is the property that actually mattered.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def regret_match(regret: Sequence[float]) -> list[float]:
    """Positive-regret-proportional strategy over a flat action vector.

    Uniform when no regret is positive (notably at t=0, where all regret is 0).
    """
    pos = [x if x > 0.0 else 0.0 for x in regret]
    s = sum(pos)
    if s > 0.0:
        return [x / s for x in pos]
    u = 1.0 / len(regret)
    return [u] * len(regret)


def regret_match_np(regret: np.ndarray, *, axis: int) -> np.ndarray:
    """Vectorized `regret_match` — normalizes along ``axis`` (the action axis).

    The action axis differs by caller: the subgame solver lays regret out as
    (action, hand) and the ReBeL trunk as (card, action), so the axis is
    explicit rather than assumed.
    """
    pos = np.maximum(regret, 0.0)
    s = pos.sum(axis=axis, keepdims=True)
    unif = 1.0 / regret.shape[axis]
    return np.where(s > 0.0, pos / np.where(s > 0.0, s, 1.0), unif)
