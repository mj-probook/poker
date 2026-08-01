"""Generate the multiway advisory artifact (solver/data/multiway_advisory.npz).

Two kinds of ADVISORY data for the tier-3 multiway drills (never grades):

  * per river board: a CERTIFIED HU-collapsed solve (uniform vs uniform at
    the multiway pot) iterated until the measured Nash gap ≤ MW_TARGET_GAP,
    certified on the float32-rounded strategies (the bytes that ship);
  * per (board, street, class): seeded Monte Carlo equity vs two uniform
    villains (n and seed formula disclosed in the drill's advisory note).

    uv run python scripts/gen_multiway_advisory.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokerlab.charts import hands  # noqa: E402
from pokerlab.drills.multiway import (DATA_PATH, MW_TARGET_GAP,  # noqa: E402
                                      build_advisory_solver, equity_seed,
                                      hero_combo, mc_equity_vs_field)
from pokerlab.drills.river import river_boards  # noqa: E402
from pokerlab.engine.cards import card_from_str  # noqa: E402

CHUNK = 400
MAX_ITERS = 20_000


def _cards(s: str) -> tuple[int, ...]:
    return tuple(card_from_str(s[i:i + 2]) for i in range(0, len(s), 2))


def _equities(board_str: str) -> np.ndarray:
    board = _cards(board_str)
    out = np.full(169, np.nan, dtype=np.float32)
    for i, cls in enumerate(hands.HAND_CLASSES):
        hero = hero_combo(cls, board)
        if hero is None:
            continue
        out[i] = mc_equity_vs_field(hero, board,
                                    seed=equity_seed(board_str, cls))
    return out


def main() -> None:
    arrays: dict[str, np.ndarray] = {}
    t0 = time.time()
    boards = river_boards()
    worst = 0.0
    for bi, (board5, tex) in enumerate(boards, 1):
        solver = build_advisory_solver(board5)
        it, gap = 0, float("inf")
        while it < MAX_ITERS:
            solver.iterate(CHUNK)
            it += CHUNK
            gap = solver.exploitability()
            if gap <= MW_TARGET_GAP:
                break
        if gap > MW_TARGET_GAP:
            raise SystemExit(f"{board5}: gap {gap:.4f}bb after {it} iters — "
                             "do not ship")
        avg32 = [a.astype(np.float32) for a in solver._avg_compressed()]
        gap = solver.exploitability(
            avg=[a.astype(np.float64) for a in avg32])
        if gap > MW_TARGET_GAP:
            raise SystemExit(f"{board5}: float32 rounding pushed the gap to "
                             f"{gap:.4f}bb — do not ship")
        for nid, a in enumerate(avg32):
            arrays[f"solve/{board5}/avg/{nid}"] = a
        arrays[f"solve/{board5}/meta"] = np.array([gap])
        worst = max(worst, gap)
        arrays[f"eq/{board5}/river"] = _equities(board5)
        arrays[f"eq/{board5[:6]}/flop"] = _equities(board5[:6])
        print(f"[{bi:2d}/{len(boards)}] {board5} ({tex:14s}) gap={gap:.4f} "
              f"iters={it} ({time.time() - t0:.0f}s)", flush=True)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    import os
    # tmp name must END in .npz — np.savez appends the suffix otherwise,
    # and the rename would find nothing (2026-08-01, measured the hard way)
    tmp = DATA_PATH.with_name(DATA_PATH.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, DATA_PATH)
    print(f"\nwrote {DATA_PATH} ({DATA_PATH.stat().st_size / 1024:.0f} KB), "
          f"worst gap {worst:.4f}bb, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
