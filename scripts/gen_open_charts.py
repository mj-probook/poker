"""Generate the checked-in open-game chart artifact (data/open_charts.npz).

Solves every drill-grid formation — OPEN_FORMATIONS × DEPTHS × ANTES, which
since the table-size axis includes every 3–8-handed table alongside 9-max
and HU — with the certified open-game solver (fold/raise/jam all priced) and
stores the full strategy set: opener frequencies + EVs, every responder's
defend strategy per threat, the opener's call-vs-re-jam strategies. The FULL
set is stored because the fast-suite cache test re-certifies each
formation's Nash gap from scratch, which needs every infoset's strategy, not
only the ones drills read. float32: plenty against a 0.1bb ε floor.

INCREMENTAL: keys already present in the shipped artifact are copied through
byte-for-byte, only missing keys are solved — honest because the fast suite
re-certifies every stored formation from scratch regardless of when it was
solved.

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

PARTS = ("sizes", "open_freq", "open_ev", "defend_freq", "defend_ev",
         "defend_fold_ev", "callback_freq", "meta")


def main() -> None:
    arrays: dict[str, np.ndarray] = {}
    existing: dict[str, np.ndarray] = {}
    if DATA_PATH.exists():
        with np.load(DATA_PATH) as d:
            existing = {name: d[name] for name in d.files}
    worst = 0.0
    t0 = time.time()
    n = len(OPEN_FORMATIONS) * len(DEPTHS) * len(ANTES)
    done = reused = 0
    for formation, (table, opener) in OPEN_FORMATIONS.items():
        behind = table[table.index(opener) + 1:]
        for depth in DEPTHS:
            for ante in ANTES:
                key = open_cache_key(formation, float(depth), float(ante))
                done += 1
                if f"{key}/meta" in existing:
                    for part in PARTS:
                        arrays[f"{key}/{part}"] = existing[f"{key}/{part}"]
                    worst = max(worst, float(existing[f"{key}/meta"][2]))
                    reused += 1
                    continue
                sol = solve_open(opener, float(depth), float(ante),
                                 table=table,
                                 sizes=sizes_for_depth(float(depth)))
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
                solve_open.cache_clear()   # 555 cached chain models = RAM
                print(f"[{done:3d}/{n}] {key:26s} "
                      f"expl={sol.exploitability:.2e} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    # atomic replace: a concurrent suite run must never read a half-
    # written zip
    import os
    # tmp name must END in .npz — np.savez appends the suffix otherwise,
    # and the rename would find nothing (2026-08-01, measured the hard way)
    tmp = DATA_PATH.with_name(DATA_PATH.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, DATA_PATH)
    size_mb = DATA_PATH.stat().st_size / 2**20
    print(f"\nwrote {DATA_PATH} ({size_mb:.1f} MB), {reused} reused, "
          f"{n - reused} solved, worst Nash gap {worst:.2e}, "
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
