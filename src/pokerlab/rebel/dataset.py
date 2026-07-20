"""Self-generated PBS→CFV training data for the Leduc value net (§3 Slice G).

Targets are the counterfactual values the full-game CFR+ solution σ* assigns to
the round-2 continuation at a public belief state — the same "leaf values from the
full CFR+ solution" the oracle harness uses, so the trained net is a drop-in for
the oracle. Each of the five round-2-entry betting lines is labelled separately,
because σ*'s round-2 play (and hence the continuation value) is line-dependent.

Sampling (the ReBeL off-blueprint-belief trick): most samples are the line's
equilibrium reach vectors PERTURBED by random multiplicative noise, so the net is
accurate exactly where the depth-limited trunk's beliefs live and drift; the rest
are broad/polarized/random ranges for global coverage. All in-house — σ* is our
own CFR+ solve, no vendor data. Reproducible under seed.
"""

from __future__ import annotations

import numpy as np

from pokerlab.cfr.cfr import CFRSolver
from pokerlab.cfr.game import build_tree
from pokerlab.cfr.leduc import LeducPoker

from .pbs import NUM_CARDS, PBS
from .resolve import leaf_reach, round2_leaf_lines
from .trunk import fixed_continuation_leaf_values


def _perturb(base: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    """Multiplicative log-normal perturbation of a reach vector, clipped to [0,1]."""
    return np.clip(base * np.exp(rng.normal(0.0, sigma, NUM_CARDS)), 0.0, 1.0)


def _broad(rng: np.random.Generator) -> np.ndarray:
    mode = rng.integers(0, 2)
    if mode == 0:                              # polarized (some hands folded out)
        r = rng.uniform(0.0, 1.0, NUM_CARDS)
        r[r < 0.4] = 0.0
        return r
    return rng.uniform(0.0, 1.0, NUM_CARDS)    # fully random


def generate_dataset(n_per_line: int = 6000, *, seed: int = 0,
                     full_iters: int = 1000, perturb_frac: float = 0.7):
    """Return (features[N,17], targets[N,12]) of PBS -> σ* continuation CFVs."""
    tree = build_tree(LeducPoker())
    solver = CFRSolver(tree, plus=True)
    solver.run(full_iters)
    sigma = solver.average_profile()
    leaf_fn = fixed_continuation_leaf_values(sigma)
    round1 = {k: v for k, v in sigma.items() if "|BNone|" in k}

    rng = np.random.default_rng(seed)
    feats, targs = [], []
    for line in round2_leaf_lines():
        contrib, bets = line["contrib"], line["bets"]
        base = leaf_reach(round1, line)        # (2,6) equilibrium reaches for the line
        for i in range(n_per_line):
            if rng.random() < perturb_frac:
                # off-blueprint beliefs around the line's equilibrium reach.
                scale = rng.uniform(0.15, 1.2)
                r0 = _perturb(base[0], rng, scale)
                r1 = _perturb(base[1], rng, scale)
            else:
                r0, r1 = _broad(rng), _broad(rng)
            if r0.sum() < 1e-6 or r1.sum() < 1e-6:
                continue
            cfv = leaf_fn(contrib, bets, r0, r1)   # (2,6) σ* continuation, line-specific
            pbs = PBS(public_card=None, bets=bets, contrib=contrib,
                      reach=np.stack([r0, r1]))
            feats.append(pbs.features())
            targs.append(cfv.reshape(-1))
    return np.asarray(feats, dtype=np.float32), np.asarray(targs, dtype=np.float32)
