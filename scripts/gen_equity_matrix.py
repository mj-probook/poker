"""Generate the 169×169 preflop all-in equity matrix (impl doc §3 Slice C).

Seeded Monte-Carlo: for every unordered pair of hand classes we deal concrete
suit-consistent hole cards (card-removal aware), a 5-card board from the rest,
and score both 7-card hands with phevaluator. Suits are randomised per trial so
the estimate averages over suit interactions between the two hands.

The result is checked in as ``src/pokerlab/charts/data/equity169.npz`` (small,
reproducible with the fixed seed). Tests load that artifact; they never call
this script. Generation is one-time:

    uv run python scripts/gen_equity_matrix.py --trials 20000 --seed 1234

Only the upper triangle (i ≤ j) is simulated; the lower triangle is filled by
the exact identity  win[j,i] = 1 - win[i,j] - tie[i,j],  tie[j,i] = tie[i,j],
so the stored matrix is showdown-consistent by construction.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from phevaluator import evaluate_cards

from pokerlab.charts import hands

DATA_PATH = Path(__file__).resolve().parents[1] / "src" / "pokerlab" / "charts" / "data" / "equity169.npz"

# suit index pairs
_PAIR_SUITS = np.array([(a, b) for a in range(4) for b in range(a + 1, 4)], dtype=np.int64)      # (6,2)
_OFF_SUITS = np.array([(a, b) for a in range(4) for b in range(4) if a != b], dtype=np.int64)     # (12,2)


def _sample_hole(rng: np.random.Generator, label: str, t: int) -> np.ndarray:
    """(t,2) concrete card ids for `t` trials of a hand class."""
    hi, lo = hands.hand_ranks(label)
    hi_i, lo_i = hi - 2, lo - 2
    if hands.is_pair(label):
        idx = rng.integers(0, 6, t)
        suits = _PAIR_SUITS[idx]                       # (t,2)
        return hi_i * 4 + suits
    if label.endswith("s"):
        s = rng.integers(0, 4, t)                      # (t,)
        return np.stack([hi_i * 4 + s, lo_i * 4 + s], axis=1)
    idx = rng.integers(0, 12, t)
    suits = _OFF_SUITS[idx]                             # (t,2)
    return np.stack([hi_i * 4 + suits[:, 0], lo_i * 4 + suits[:, 1]], axis=1)


def _resolve_collisions(rng, label_j, hi_cards, hj_cards):
    """Resample hand-j rows that share a card with hand-i (rank overlap cases)."""
    for _ in range(500):
        clash = (
            (hj_cards[:, 0:1] == hi_cards).any(axis=1)
            | (hj_cards[:, 1:2] == hi_cards).any(axis=1)
        )
        n = int(clash.sum())
        if n == 0:
            return hj_cards
        hj_cards[clash] = _sample_hole(rng, label_j, n)
    raise RuntimeError(f"could not deconflict {label_j} after 500 resamples")


def _deal_boards(rng: np.random.Generator, holes: np.ndarray, t: int) -> np.ndarray:
    """(t,5) board cards disjoint from the (t,4) `holes` via random-key selection."""
    keys = rng.random((t, 52))
    rows = np.arange(t)[:, None]
    keys[rows, holes] = np.inf                          # holes never chosen
    part = np.argpartition(keys, 5, axis=1)[:, :5]
    return part


def _cell_equity(rng, label_i, label_j, trials):
    hi = _sample_hole(rng, label_i, trials)
    hj = _sample_hole(rng, label_j, trials)
    hj = _resolve_collisions(rng, label_j, hi, hj)
    holes = np.concatenate([hi, hj], axis=1)            # (t,4)
    board = _deal_boards(rng, holes, trials)            # (t,5)
    seven_i = np.concatenate([hi, board], axis=1).tolist()
    seven_j = np.concatenate([hj, board], axis=1).tolist()
    ri = np.fromiter((evaluate_cards(*row) for row in seven_i), dtype=np.int32, count=trials)
    rj = np.fromiter((evaluate_cards(*row) for row in seven_j), dtype=np.int32, count=trials)
    win = float(np.mean(ri < rj))   # smaller rank = stronger
    tie = float(np.mean(ri == rj))
    return win, tie


def generate(trials: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(hands.HAND_CLASSES)
    win = np.zeros((n, n), dtype=np.float64)
    tie = np.zeros((n, n), dtype=np.float64)
    rng = np.random.default_rng(seed)
    start = time.time()
    cells = n * (n + 1) // 2
    done = 0
    for i in range(n):
        for j in range(i, n):
            w, t = _cell_equity(rng, hands.HAND_CLASSES[i], hands.HAND_CLASSES[j], trials)
            win[i, j], tie[i, j] = w, t
            if i != j:
                tie[j, i] = t
                win[j, i] = 1.0 - w - t
            done += 1
        if (i + 1) % 13 == 0:
            elapsed = time.time() - start
            print(f"  row {i + 1}/{n}  cells {done}/{cells}  {elapsed:.1f}s")
    return win.astype(np.float32), tie.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=Path, default=DATA_PATH)
    args = ap.parse_args()

    print(f"generating 169×169 equity matrix: trials={args.trials} seed={args.seed}")
    win, tie = generate(args.trials, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        win=win,
        tie=tie,
        labels=np.array(hands.HAND_CLASSES),
        trials=np.int64(args.trials),
        seed=np.int64(args.seed),
    )
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1024:.0f} KiB)")
    aa, kk = hands.HAND_INDEX["AA"], hands.HAND_INDEX["KK"]
    print(f"sanity: AA vs KK win+tie/2 = {win[aa, kk] + tie[aa, kk] / 2:.4f} (expect ~0.82)")


if __name__ == "__main__":
    main()
