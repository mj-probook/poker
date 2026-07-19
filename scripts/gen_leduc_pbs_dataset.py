#!/usr/bin/env python
"""Generate the Leduc PBS→CFV value-net dataset (plan §7 L3; impl doc §3 Slice G).

Self-generated, ToS-clean: labels are the round-2 subgame's equilibrium
counterfactual values from our own CFR+ solver, sampled at reach-perturbed
(off-equilibrium) public belief states per ReBeL. Reproducible under --seed.

Cached to artifacts/ (gitignored). Reproduce with:

    uv run python scripts/gen_leduc_pbs_dataset.py --n-per-state 400 --seed 0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pokerlab.rebel.dataset import generate_dataset

DEFAULT_OUT = Path("artifacts/leduc_pbs_dataset.npz")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-per-state", type=int, default=400,
                    help="samples per pot level (3 levels)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full-iters", type=int, default=1000,
                    help="CFR+ iterations for the σ* blueprint")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    x, y = generate_dataset(n_per_state=args.n_per_state, seed=args.seed,
                            full_iters=args.full_iters)
    np.savez_compressed(args.out, features=x, targets=y)
    print(f"wrote {args.out}  features={x.shape} targets={y.shape}  "
          f"in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
