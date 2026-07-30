"""Generate the checked-in open-game chart artifact (data/open_charts.npz).

Solves every drill-grid formation — OPEN_FORMATIONS × DEPTHS × ANTES — with
the certified open-game solver (fold/raise/jam all priced) and stores the
full strategy set: opener frequencies + EVs, every responder's defend
strategy per threat, the opener's call-vs-re-jam strategies. The FULL set is
stored because the fast-suite cache test re-certifies each formation's Nash
gap from scratch, which needs every infoset's strategy, not only the ones
drills read. float32: plenty against a 0.1bb ε floor, half the bytes.

    uv run python scripts/gen_open_charts.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokerlab.charts.openraise import (DATA_PATH, OPEN_FORMATIONS,  # noqa: E402
                                       _rlabel, open_cache_key,
                                       sizes_for_depth, solve_open)
from pokerlab.drills.categories import ANTES, DEPTHS  # noqa: E402


def main() -> None:
    arrays: dict[str, np.ndarray] = {}
    worst = 0.0
    t0 = time.time()
    n = len(OPEN_FORMATIONS) * len(DEPTHS) * len(ANTES)
    done = 0
    for formation, (table, opener) in OPEN_FORMATIONS.items():
        behind = table[table.index(opener) + 1:]
        for depth in DEPTHS:
            for ante in ANTES:
                sol = solve_open(opener, float(depth), float(ante),
                                 table=table,
                                 sizes=sizes_for_depth(float(depth)))
                key = open_cache_key(formation, float(depth), float(ante))
                threats = ["jam"] + [_rlabel(r) for r in sol.sizes]
                labels = threats + ["fold"]
                f32 = np.float32
                arrays[f"{key}/sizes"] = np.array(sol.sizes, dtype=f32)
                arrays[f"{key}/open_freq"] = np.stack(
                    [sol.open_freq[lab] for lab in labels]).astype(f32)
                arrays[f"{key}/open_ev"] = np.stack(
                    [sol.open_ev[lab] for lab in labels]).astype(f32)
                arrays[f"{key}/defend_freq"] = np.stack(
                    [[sol.defend_freq[(q, t)] for q in behind]
                     for t in threats]).astype(f32)
                arrays[f"{key}/defend_ev"] = np.stack(
                    [[sol.defend_ev[(q, t)] for q in behind]
                     for t in threats]).astype(f32)
                arrays[f"{key}/defend_fold_ev"] = np.array(
                    [[sol.defend_fold_ev[(q, t)] for q in behind]
                     for t in threats], dtype=f32)
                arrays[f"{key}/callback_freq"] = (
                    np.stack([[sol.callback_freq[(t, q)] for q in behind]
                              for t in threats[1:]]).astype(f32)
                    if len(threats) > 1
                    else np.zeros((0, len(behind), 169), dtype=f32))
                arrays[f"{key}/meta"] = np.array(
                    [sol.depth_bb, sol.ante, sol.exploitability])
                worst = max(worst, sol.exploitability)
                done += 1
                print(f"[{done:3d}/{n}] {key:18s} "
                      f"expl={sol.exploitability:.2e} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA_PATH, **arrays)
    size_mb = DATA_PATH.stat().st_size / 2**20
    print(f"\nwrote {DATA_PATH} ({size_mb:.1f} MB), "
          f"worst Nash gap {worst:.2e}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
