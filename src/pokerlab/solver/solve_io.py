"""Solve-file writer + solution_index helper (impl doc §2, §3 Slice D).

A solved subgame is persisted as a JSON solve file under ``solves/`` (one file
per SpotKey), and a row is written to the ``solution_index`` table (schema in
store/schema.sql) so tier-2 grading can find it by SpotKey. The solve file holds
the root strategy per hand-class (the tier-2 lookup payload) plus provenance and
the measured exploitability.

The store wrapper (`store.db`) is Slice E's; this module only *uses*
`store.db.connect` for the schema and does a single narrow insert into
`solution_index` — it does not own or extend the store layer.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from pokerlab.solver.adapter import root_solution_by_class
from pokerlab.solver.subgame import SOLVER_VERSION, SubgameSolver

SOLVES_DIR = Path(__file__).resolve().parents[3] / "solves"


def write_solve(solver: SubgameSolver, spot_key: str, path: str | Path | None = None,
                player: int = 0, solver_version: str = SOLVER_VERSION) -> tuple[Path, float]:
    """Write a solve file for ``solver`` and return (path, exploitability_bb).

    The file name defaults to ``solves/<sanitized spot_key>.json``.
    """
    expl = solver.exploitability()
    range_ctx = f"{spot_key}|{solver_version}|expl_bb={expl:.5g}"
    solutions = root_solution_by_class(solver, player=player, range_ctx=range_ctx)
    payload = {
        "spot_key": spot_key,
        "solver_version": solver_version,
        "source": "subgame_solver",
        "board": [int(c) for c in solver.board],
        "pot_bb": solver.pot0,
        "exploitability_bb": expl,
        "exploitability_pct": expl / solver.pot0,
        "iterations": solver._t,
        "root_player": player,
        "solutions": {cls: asdict(sol) for cls, sol in solutions.items()},
    }
    if path is None:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in spot_key)
        SOLVES_DIR.mkdir(parents=True, exist_ok=True)
        path = SOLVES_DIR / f"{safe}.json"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path, expl


def load_solve(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def insert_solution_index(conn: sqlite3.Connection, spot_key: str, path: str | Path,
                          solver_version: str, exploitability: float) -> None:
    """Insert/replace a ``solution_index`` row. ``conn`` is a schema-applied
    connection (use ``store.db.connect``). Narrow local helper — the store layer
    is Slice E's and is not extended here."""
    conn.execute(
        "INSERT INTO solution_index(spot_key, path, solver_version, exploitability)"
        " VALUES (?, ?, ?, ?)"
        " ON CONFLICT(spot_key) DO UPDATE SET"
        "   path=excluded.path, solver_version=excluded.solver_version,"
        "   exploitability=excluded.exploitability",
        (spot_key, str(path), solver_version, float(exploitability)),
    )
    conn.commit()
