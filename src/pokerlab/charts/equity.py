"""Preflop all-in equity matrix accessor (impl doc §3 Slice C).

Loads the checked-in ``data/equity169.npz`` artifact (built by
``scripts/gen_equity_matrix.py``) and exposes hand-class-vs-hand-class showdown
probabilities. Chip equity of A vs B = P(A wins) + ½·P(tie).

The stored matrix is showdown-consistent by construction:
    win[i,j] + win[j,i] + tie[i,j] = 1   and   equity[i,j] + equity[j,i] = 1.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from . import hands

DATA_PATH = Path(__file__).resolve().parent / "data" / "equity169.npz"


class EquityMatrix:
    """169×169 win/tie tables indexed by canonical hand class."""

    def __init__(self, win: np.ndarray, tie: np.ndarray, labels: list[str]):
        if list(labels) != hands.HAND_CLASSES:
            raise ValueError("equity artifact labels differ from HAND_CLASSES")
        self.win = win.astype(np.float64)
        self.tie = tie.astype(np.float64)
        self.labels = list(labels)

    @property
    def equity_matrix(self) -> np.ndarray:
        """Chip-equity matrix E[i,j] = P(i beats j) + ½·P(tie)."""
        return self.win + 0.5 * self.tie

    def equity(self, a: str, b: str) -> float:
        i, j = hands.HAND_INDEX[a], hands.HAND_INDEX[b]
        return float(self.win[i, j] + 0.5 * self.tie[i, j])

    def win_tie(self, a: str, b: str) -> tuple[float, float, float]:
        i, j = hands.HAND_INDEX[a], hands.HAND_INDEX[b]
        w, t = float(self.win[i, j]), float(self.tie[i, j])
        return w, t, 1.0 - w - t


@lru_cache(maxsize=None)
def load_equity_matrix(path: str | None = None) -> EquityMatrix:
    p = Path(path) if path else DATA_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"equity matrix artifact missing: {p}\n"
            "generate it with: uv run python scripts/gen_equity_matrix.py"
        )
    with np.load(p, allow_pickle=True) as d:
        return EquityMatrix(d["win"], d["tie"], list(d["labels"]))
