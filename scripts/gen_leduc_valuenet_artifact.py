#!/usr/bin/env python
"""Generate the checked-in Leduc value-net artifact (M5 fast-CI exit).

The M5 exit — "depth-limited + value net reaches the bar on Leduc" — was only
ever asserted by slow-gated tests that train from scratch, so the fast suite
proved nothing about it (round-1 finding [1]). Training in CI is far too slow,
so we check in a small deterministic *trained net* and let the fast suite assert
the exit against it.

This script regenerates that artifact. It is fully seeded (dataset sampling,
train/val split, weight init, batch order) and CPU-only, so re-running it
reproduces the same weights bit-for-bit.

    uv run python scripts/gen_leduc_valuenet_artifact.py

Writes tests/fixtures/leduc_valuenet.pt (~25KB) and prints the numbers the fast
test then asserts. Bump N_PER_LINE / EPOCHS here if the bar ever tightens; the
committed artifact and the printed |Δ| must be regenerated together.
"""

from __future__ import annotations

from pathlib import Path

import torch

from pokerlab.cfr.cfr import CFRSolver
from pokerlab.cfr.exploit import on_policy_values
from pokerlab.cfr.game import build_tree
from pokerlab.cfr.leduc import LeducPoker
from pokerlab.rebel.dataset import generate_dataset
from pokerlab.rebel.depth_limited import DepthLimitedLeducOracle
from pokerlab.rebel.trunk import DepthLimitedSolver
from pokerlab.rebel.valuenet import net_leaf_value_fn, train_valuenet

# Pinned generation config — the fast test's bar is only meaningful next to it.
SEED = 0
N_PER_LINE = 4000
EPOCHS = 300
HIDDEN = 64
FULL_ITERS = 700          # matches tests/conftest.py's leduc_sigma_star
TRUNK_ITERS = 600

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "leduc_valuenet.pt"


def main() -> None:
    x, y = generate_dataset(n_per_line=N_PER_LINE, seed=SEED)
    net, rep = train_valuenet(x, y, hidden=HIDDEN, epochs=EPOCHS, seed=SEED,
                              device="cpu")
    print(f"[train] {x.shape[0]} samples, val_rmse={rep.val_rmse:.5f}")

    full = build_tree(LeducPoker())
    solver = CFRSolver(full, plus=True)
    solver.run(FULL_ITERS)
    sigma = solver.average_profile()
    v_full = on_policy_values(full, sigma)[0]

    trunk = DepthLimitedSolver(net_leaf_value_fn(net))
    trunk.run(TRUNK_ITERS)
    v_net = on_policy_values(build_tree(DepthLimitedLeducOracle(sigma)),
                             trunk.round1_profile())[0]
    delta = abs(v_net - v_full)
    print(f"[exit ] root value net={v_net:.6f} full={v_full:.6f} |Δ|={delta:.2e} "
          f"({'PASS' if delta <= 2e-3 else 'FAIL'} vs 2e-3)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "hidden": HIDDEN,
                "seed": SEED, "n_per_line": N_PER_LINE, "epochs": EPOCHS,
                "val_rmse": rep.val_rmse}, OUT)
    print(f"[write] {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
