"""SQLite wrapper for the checked-in schema (impl doc §2; plan §3 storage).

This module owns exactly two things: applying `store/schema.sql` to a fresh
connection, and typed insert/query helpers for the tables Slice E writes
(`drill_attempts`, `sr_state`) plus `gradings` (so the tier-3 honesty trigger
is reachable from Python). Every *derived* metric — leak rankings, accuracy
trends — is a query in `store/views.py`, never a table here (plan §3).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def schema_sql() -> str:
    return SCHEMA_PATH.read_text()


def connect(path: str | Path = ":memory:", *, check_same_thread: bool = True
            ) -> sqlite3.Connection:
    """Open a connection with the schema applied and dict-like row access."""
    conn = sqlite3.connect(str(path), check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.executescript(schema_sql())
    conn.commit()
    return conn


# --------------------------------------------------------------------------- #
# drill_attempts
# --------------------------------------------------------------------------- #
def insert_drill_attempt(conn: sqlite3.Connection, spot_key: str, kind: str,
                         chosen: str, correct: bool, ev_loss: float,
                         ts: str) -> int:
    cur = conn.execute(
        "INSERT INTO drill_attempts(spot_key, kind, chosen, correct, ev_loss, ts)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (spot_key, kind, chosen, int(bool(correct)), float(ev_loss), ts),
    )
    conn.commit()
    return int(cur.lastrowid)


def drill_attempts(conn: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM drill_attempts ORDER BY id")]


# --------------------------------------------------------------------------- #
# sr_state (SM-2 spaced repetition)
# --------------------------------------------------------------------------- #
def upsert_sr_state(conn: sqlite3.Connection, leak_key: str, easiness: float,
                    interval_days: float, reps: int, due: str) -> None:
    conn.execute(
        "INSERT INTO sr_state(leak_key, easiness, interval_days, reps, due)"
        " VALUES (?, ?, ?, ?, ?)"
        " ON CONFLICT(leak_key) DO UPDATE SET"
        "   easiness=excluded.easiness, interval_days=excluded.interval_days,"
        "   reps=excluded.reps, due=excluded.due",
        (leak_key, float(easiness), float(interval_days), int(reps), due),
    )
    conn.commit()


def get_sr_state(conn: sqlite3.Connection, leak_key: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM sr_state WHERE leak_key=?", (leak_key,)).fetchone()
    return dict(row) if row is not None else None


def all_sr_state(conn: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT * FROM sr_state ORDER BY due")]


# --------------------------------------------------------------------------- #
# gradings (HH pipeline table; here only to exercise the tier-3 trigger)
# --------------------------------------------------------------------------- #
def insert_grading(conn: sqlite3.Connection, hand_id: int, decision_idx: int,
                   tier: int, chosen: str, best: str, ev_loss: float | None,
                   leak_key: str, graded_at: str) -> int:
    cur = conn.execute(
        "INSERT INTO gradings(hand_id, decision_idx, tier, chosen, best,"
        " ev_loss, leak_key, graded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (hand_id, decision_idx, tier, chosen, best,
         None if ev_loss is None else float(ev_loss), leak_key, graded_at),
    )
    conn.commit()
    return int(cur.lastrowid)
