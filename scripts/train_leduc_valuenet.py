#!/usr/bin/env python
"""Train + evaluate the Leduc PBS value net end-to-end (plan §7 L3; §3 Slice G).

Reproducible pipeline: generate self-generated PBS→CFV data (or load the cache) →
train the MLP → report the numbers that matter for M5:

  * value-net held-out RMSE (does the net learn the PBS value function?)
  * net-driven depth-limited trunk: root-value error vs the full CFR+ solution
    (does the net's value replace the oracle in the trunk?)
  * ORACLE-leaf safe-resolve NashConv (the M5 ≤2e-3 bar, with leaf values from
    the full CFR+ solution + the CFR-D safe-resolving gadget)

Everything is in-house and seeded. Run:

    uv run python scripts/train_leduc_valuenet.py --n-per-state 400 --epochs 400
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pokerlab.cfr.cfr import CFRSolver
from pokerlab.cfr.exploit import nash_conv, on_policy_values
from pokerlab.cfr.game import build_tree
from pokerlab.cfr.leduc import LeducPoker
from pokerlab.rebel.dataset import generate_dataset
from pokerlab.rebel.depth_limited import DepthLimitedLeducOracle
from pokerlab.rebel.safe_resolve import safe_rebel_agent_profile
from pokerlab.rebel.trunk import DepthLimitedSolver, fixed_continuation_leaf_values
from pokerlab.rebel.valuenet import net_leaf_value_fn, train_valuenet

DEFAULT_DATA = Path("artifacts/leduc_pbs_dataset.npz")


def _load_or_generate(path: Path, n_per_line: int, seed: int, regen: bool):
    if path.exists() and not regen:
        d = np.load(path)
        return d["features"], d["targets"]
    x, y = generate_dataset(n_per_line=n_per_line, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=x, targets=y)
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-per-line", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu (deterministic) or mps")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--regen", action="store_true", help="regenerate the dataset")
    ap.add_argument("--full-iters", type=int, default=1200)
    ap.add_argument("--trunk-iters", type=int, default=600)
    ap.add_argument("--resolve-iters", type=int, default=600)
    args = ap.parse_args()

    t0 = time.time()
    x, y = _load_or_generate(args.dataset, args.n_per_line, args.seed, args.regen)
    print(f"[data] {x.shape[0]} PBS→CFV samples ({time.time() - t0:.1f}s)")

    net, rep = train_valuenet(x, y, epochs=args.epochs, seed=args.seed,
                              device=args.device)
    print(f"[train] {rep.epochs} epochs on {rep.device}: "
          f"final_loss={rep.final_loss:.2e}  val_rmse={rep.val_rmse:.4f}")

    # Full-game reference.
    full = build_tree(LeducPoker())
    solver = CFRSolver(full, plus=True)
    solver.run(args.full_iters)
    sigma = solver.average_profile()
    v_full = on_policy_values(full, sigma)[0]
    print(f"[ref ] full CFR+ NashConv={nash_conv(full, sigma):.2e}  root_value={v_full:.4f}")

    # Amended-A exit: net-driven depth-limited trunk reproduces the root value.
    net_trunk = DepthLimitedSolver(net_leaf_value_fn(net))
    net_trunk.run(args.trunk_iters)
    round1_net = net_trunk.round1_profile()
    dl_tree = build_tree(DepthLimitedLeducOracle(sigma))
    v_net = on_policy_values(dl_tree, round1_net)[0]
    d_root = abs(v_net - v_full)
    print(f"[net ] net-driven trunk root_value={v_net:.4f}  |Δ|={d_root:.4f}  "
          f"({'PASS' if d_root <= 2e-3 else 'above'} 2e-3)")

    # Net-driven trunk + safe-resolve full agent (informational: bounded by the
    # spurious-round-1 limit of depth-limited solving — see docs/notes).
    nc_net = nash_conv(full, safe_rebel_agent_profile(
        round1_net, net_leaf_value_fn(net), resolve_iters=args.resolve_iters))
    print(f"[net ] net-driven + safe-resolve full-agent NashConv = {nc_net:.5f}")

    # Oracle bar: leaves from the full CFR+ solution + safe re-solving.
    round1 = {k: v for k, v in sigma.items() if "|BNone|" in k}
    nc = nash_conv(full, safe_rebel_agent_profile(
        round1, fixed_continuation_leaf_values(sigma), resolve_iters=args.resolve_iters))
    print(f"[M5  ] oracle-leaf + safe-resolve NashConv = {nc:.6f}  "
          f"({'PASS' if nc <= 2e-3 else 'above'} 2e-3)")
    print(f"[done] {time.time() - t0:.1f}s total")


if __name__ == "__main__":
    main()
