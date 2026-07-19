"""Population action-frequency table for tier-3 grading (Slice F, impl doc §3).

Loads a versioned in-house baseline (`population/default.toml`) mapping each
``"<formation>|<street>"`` spot to a normalized distribution over action_types.
Tier-3 grading uses this — and ONLY this — as its reference: no EV is ever
derived here (grading-honesty hard rule), just how often the pool takes an
action so deviations can be flagged.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "population" / "default.toml"


@dataclass(frozen=True)
class Population:
    version: str
    spots: dict[str, dict[str, float]]  # "formation|street" -> {action_type: freq}

    def frequency(self, formation: str, street: str, action_type: str) -> float | None:
        """Population frequency of ``action_type`` in this spot, or None if the
        spot has no baseline."""
        dist = self.spots.get(f"{formation}|{street}")
        if dist is None:
            return None
        return float(dist.get(action_type, 0.0))


def load_population(path: str | Path | None = None) -> Population:
    p = Path(path) if path is not None else _DEFAULT_PATH
    data = tomllib.loads(p.read_text())
    version = str(data.pop("version", "unknown"))
    spots = {k: {a: float(v) for a, v in dist.items()} for k, dist in data.items()}
    return Population(version=version, spots=spots)
