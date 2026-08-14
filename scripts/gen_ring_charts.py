"""Generate the checked-in ring chart artifact (charts/data/ring_charts.npz).

Solves every drill-grid formation — jammers × DEPTHS × ANTES at every
TABLE SIZE 3–9 (2 is the HU jamfold module's game) — with the certified
chain solver and stores strategies + per-hand EVs. Deterministic and 100%
self-generated, like the equity matrix.

INCREMENTAL: keys already present in the shipped artifact are copied through
byte-for-byte, only missing keys are solved. Honest because provenance does
not matter here — the cache test re-certifies every stored formation's Nash
gap from scratch on every suite run, whenever it was solved.

    uv run python scripts/gen_ring_charts.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokerlab.charts.ring import (DATA_PATH, RING_JAMMERS, cache_key,  # noqa: E402
                                  solve_ring, table_for_size)
from pokerlab.drills.categories import ANTES, DEPTHS  # noqa: E402


def formations() -> list[tuple[str, int, tuple[str, ...]]]:
    """[(jammer, table_size, table)] for the whole drill grid."""
    out = [(j, 9, table_for_size(9)) for j in RING_JAMMERS]
    for n in (3, 4, 5, 6, 7, 8):
        table = table_for_size(n)
        out.extend((j, n, table) for j in table[:-1])   # every non-BB seat
    return out


def main() -> None:
    arrays: dict[str, np.ndarray] = {}
    existing: dict[str, np.ndarray] = {}
    if DATA_PATH.exists():
        with np.load(DATA_PATH) as d:
            existing = {name: d[name] for name in d.files}
    worst = 0.0
    t0 = time.time()
    fms = formations()
    total = len(fms) * len(DEPTHS) * len(ANTES)
    done = reused = 0
    for jammer, size, table in fms:
        behind = table[table.index(jammer) + 1:]
        for depth in DEPTHS:
            for ante in ANTES:
                key = cache_key(jammer, float(depth), float(ante), size)
                done += 1
                if f"{key}/meta" in existing:
                    for part in ("jam", "jam_ev", "calls", "call_ev",
                                 "call_fold", "meta"):
                        arrays[f"{key}/{part}"] = existing[f"{key}/{part}"]
                    worst = max(worst, float(existing[f"{key}/meta"][3]))
                    reused += 1
                    continue
                sol = solve_ring(jammer, float(depth), float(ante),
                                 table=table)
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
                print(f"[{done:3d}/{total}] {key:22s} "
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
    size_kb = DATA_PATH.stat().st_size / 1024
    print(f"\nwrote {DATA_PATH} ({size_kb:.0f} KB), {reused} reused, "
          f"{total - reused} solved, worst Nash gap {worst:.2e}, "
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
