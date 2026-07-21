"""Generate the checked-in ring chart artifact (charts/data/ring_charts.npz).

Solves every drill-grid formation — RING_JAMMERS × DEPTHS × ANTES — with the
certified chain solver and stores strategies + per-hand EVs. Deterministic and
100% self-generated, like the equity matrix. Re-run only on an intentional
model change; the cache tests re-certify the shipped artifact on every suite
run regardless.

    uv run python scripts/gen_ring_charts.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokerlab.charts.ring import (RING_JAMMERS, RING_ORDER, cache_key,  # noqa: E402
                                  DATA_PATH, solve_ring)
from pokerlab.drills.categories import ANTES, DEPTHS  # noqa: E402


def main() -> None:
    arrays: dict[str, np.ndarray] = {}
    worst = 0.0
    t0 = time.time()
    n = len(RING_JAMMERS) * len(DEPTHS) * len(ANTES)
    done = 0
    for jammer in RING_JAMMERS:
        behind = RING_ORDER[RING_ORDER.index(jammer) + 1:]
        for depth in DEPTHS:
            for ante in ANTES:
                sol = solve_ring(jammer, float(depth), float(ante))
                key = cache_key(jammer, float(depth), float(ante))
                arrays[f"{key}/jam"] = sol.jam
                arrays[f"{key}/jam_ev"] = sol.jam_ev
                arrays[f"{key}/calls"] = np.stack([sol.calls[q] for q in behind])
                arrays[f"{key}/call_ev"] = np.stack(
                    [sol.call_ev[q] for q in behind])
                arrays[f"{key}/call_fold"] = np.array(
                    [sol.call_fold_ev[q] for q in behind])
                arrays[f"{key}/meta"] = np.array(
                    [sol.depth_bb, sol.ante, sol.jam_fold_ev,
                     sol.exploitability])
                worst = max(worst, sol.exploitability)
                done += 1
                print(f"[{done:3d}/{n}] {key:18s} "
                      f"expl={sol.exploitability:.2e}", flush=True)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA_PATH, **arrays)
    size_kb = DATA_PATH.stat().st_size / 1024
    print(f"\nwrote {DATA_PATH} ({size_kb:.0f} KB), "
          f"worst Nash gap {worst:.2e}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
