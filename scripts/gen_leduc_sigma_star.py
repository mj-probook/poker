#!/usr/bin/env python
"""Generate the checked-in converged Leduc CFR+ profile (fast-suite fixture).

`leduc_sigma_star` is a near-equilibrium Leduc profile that nine Slice-G ReBeL
tests build on. It was already session-scoped, so the cost was paid once — but
once was still 4.0s of the fast suite's 60s budget, spent recomputing a fully
deterministic artifact on every run.

So we check it in, the same way the M5 value-net artifact is checked in. 700
CFR+ iterations on Leduc is bit-for-bit reproducible (no RNG anywhere in the
path), so the artifact is a pure function of this script.

    uv run python scripts/gen_leduc_sigma_star.py

Writes tests/fixtures/leduc_sigma_star.json (~78KB) and prints the NashConv the
fixture then re-checks on load. The tree is NOT stored — rebuilding it is 0.01s
and storing it would let the profile and the tree drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

from pokerlab.cfr.cfr import CFRSolver
from pokerlab.cfr.exploit import nash_conv
from pokerlab.cfr.game import build_tree
from pokerlab.cfr.leduc import LeducPoker

ITERS = 700
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "leduc_sigma_star.json"


def main() -> None:
    tree = build_tree(LeducPoker())
    solver = CFRSolver(tree, plus=True)
    solver.run(ITERS)
    profile = solver.average_profile()

    nc = nash_conv(tree, profile)
    print(f"CFR+ {ITERS} iters -> NashConv = {nc:.6g}")
    if nc >= 1e-3:
        raise SystemExit(f"refusing to write: NashConv {nc:.6g} misses the 1e-3 bar")

    # action keys are ints; JSON stringifies them, so the loader casts back
    payload = {
        "iters": ITERS,
        "nash_conv": nc,
        "profile": {k: {str(a): p for a, p in d.items()} for k, d in profile.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024}KB, {len(profile)} infosets)")


if __name__ == "__main__":
    main()
