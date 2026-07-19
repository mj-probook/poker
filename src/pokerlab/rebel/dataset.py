"""Self-generated PBS→CFV training data for the Leduc value net (§3 Slice G).

Targets are the counterfactual values the full-game CFR+ solution σ* assigns to
the round-2 continuation at a public belief state — the same "leaf values from the
full CFR+ solution" the oracle harness uses, so the trained net is a drop-in for
the oracle. We sample reach vectors perturbed away from equilibrium (per ReBeL)
so the net sees off-equilibrium beliefs across the simplex; each is labelled by
evaluating σ*'s continuation at that belief (belief-linear, cheap and exact).

All in-house — σ* is our own CFR+ solve, no vendor data. Reproducible under seed.
The round-2 value depends only on (pot, belief), so the three pot levels are
sampled once each with a canonical betting label.
"""

from __future__ import annotations

import numpy as np

from pokerlab.cfr.cfr import CFRSolver
from pokerlab.cfr.game import build_tree
from pokerlab.cfr.leduc import LeducPoker

from .pbs import NUM_CARDS, PBS
from .trunk import fixed_continuation_leaf_values

# (contrib, canonical bets) for each round-2-entry pot level.
_POT_STATES = [
    ((1, 1), (1, 1)),         # check-check
    ((3, 3), (2, 1)),         # raise-call
    ((5, 5), (2, 2, 1)),      # raise-raise-call
]


def _sample_reach(rng: np.random.Generator) -> np.ndarray:
    """A reach vector over 6 cards in [0,1], biased to cover the simplex edges."""
    mode = rng.integers(0, 3)
    if mode == 0:                      # broad/uniform-ish ranges
        return rng.uniform(0.2, 1.0, NUM_CARDS)
    if mode == 1:                      # polarized (some hands folded out)
        r = rng.uniform(0.0, 1.0, NUM_CARDS)
        r[r < 0.4] = 0.0
        return r
    return rng.uniform(0.0, 1.0, NUM_CARDS)  # fully random


def generate_dataset(n_per_state: int = 400, *, seed: int = 0,
                     full_iters: int = 1000):
    """Return (features[N,15], targets[N,12]) of PBS -> σ* continuation CFVs."""
    tree = build_tree(LeducPoker())
    solver = CFRSolver(tree, plus=True)
    solver.run(full_iters)
    leaf_fn = fixed_continuation_leaf_values(solver.average_profile())

    rng = np.random.default_rng(seed)
    feats, targs = [], []
    for contrib, bets in _POT_STATES:
        for _ in range(n_per_state):
            r0, r1 = _sample_reach(rng), _sample_reach(rng)
            if r0.sum() < 1e-6 or r1.sum() < 1e-6:
                continue
            cfv = leaf_fn(contrib, bets, r0, r1)     # (2,6) σ* continuation
            pbs = PBS(public_card=None, bets=bets, contrib=contrib,
                      reach=np.stack([r0, r1]))
            feats.append(pbs.features())
            targs.append(cfv.reshape(-1))
    return np.asarray(feats, dtype=np.float32), np.asarray(targs, dtype=np.float32)
