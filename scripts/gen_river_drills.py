"""Generate the checked-in river-strategy artifact (solver/data/).

Solves every `river_boards()` board (flops25 flops + deterministic runouts,
up to two per texture) with the M3 subgame solver on the 33/75/jam grid,
iterating in chunks until the MEASURED Nash gap is <= TARGET_GAP_BB. Stores
the compressed (live-combo) average strategy of every decision node, because
the fast-suite test re-certifies each board's gap from scratch via
`exploitability(avg=stored)` — the artifact ships with its certificate.

    uv run python scripts/gen_river_drills.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokerlab.drills.river import (DATA_PATH, TARGET_GAP_BB,  # noqa: E402
                                   build_river_solver, river_boards)

CHUNK = 400
MAX_ITERS = 20_000


def main() -> None:
    arrays: dict[str, np.ndarray] = {}
    boards = river_boards()
    worst = 0.0
    t0 = time.time()
    for i, (board, tex) in enumerate(boards, 1):
        solver = build_river_solver(board)
        it, gap = 0, float("inf")
        while it < MAX_ITERS:
            solver.iterate(CHUNK)
            it += CHUNK
            gap = solver.exploitability()
            if gap <= TARGET_GAP_BB:
                break
        if gap > TARGET_GAP_BB:
            raise SystemExit(f"{board} failed to converge: gap={gap:.4f}bb "
                             f"after {it} iters — do not ship this artifact")
        # certify the ROUNDED strategies — the exact bytes that ship — so the
        # stored gap is the one the fast suite will re-measure
        avg32 = [a.astype(np.float32) for a in solver._avg_compressed()]
        gap = solver.exploitability(avg=[a.astype(np.float64) for a in avg32])
        if gap > TARGET_GAP_BB:
            raise SystemExit(f"{board}: float32 rounding pushed the gap to "
                             f"{gap:.4f}bb — do not ship this artifact")
        for nid, a in enumerate(avg32):
            arrays[f"{board}/avg/{nid}"] = a
        arrays[f"{board}/meta"] = np.array([gap], dtype=np.float64)
        worst = max(worst, gap)
        print(f"[{i:2d}/{len(boards)}] {board} ({tex:12s}) "
              f"gap={gap:.5f}bb iters={it} ({time.time() - t0:.0f}s)",
              flush=True)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA_PATH, **arrays)
    size_kb = DATA_PATH.stat().st_size / 1024
    print(f"\nwrote {DATA_PATH} ({size_kb:.0f} KB), "
          f"worst gap {worst:.5f}bb, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
