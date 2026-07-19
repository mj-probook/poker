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


def gradings(conn: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT * FROM gradings ORDER BY id")]


# --------------------------------------------------------------------------- #
# imported_hands (Slice F: raw HH + parsed summary, so the batch worker can
# re-derive a decision from storage when a solve finally lands)
# --------------------------------------------------------------------------- #
def insert_imported_hand(conn: sqlite3.Connection, site: str, raw: str,
                         parsed_json: str, imported_at: str) -> int:
    cur = conn.execute(
        "INSERT INTO imported_hands(site, raw, parsed_json, imported_at)"
        " VALUES (?, ?, ?, ?)",
        (site, raw, parsed_json, imported_at),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_imported_hand(conn: sqlite3.Connection, hand_id: int) -> dict | None:
    row = conn.execute(
        "SELECT * FROM imported_hands WHERE id=?", (hand_id,)).fetchone()
    return dict(row) if row is not None else None


# --------------------------------------------------------------------------- #
# solution_index (tier-2 library: a spot_key hit means we can grade exactly)
# --------------------------------------------------------------------------- #
def index_solution(conn: sqlite3.Connection, spot_key: str, path: str,
                   solver_version: str, exploitability: float) -> None:
    conn.execute(
        "INSERT INTO solution_index(spot_key, path, solver_version, exploitability)"
        " VALUES (?, ?, ?, ?)"
        " ON CONFLICT(spot_key) DO UPDATE SET path=excluded.path,"
        "   solver_version=excluded.solver_version,"
        "   exploitability=excluded.exploitability",
        (spot_key, path, solver_version, float(exploitability)),
    )
    conn.commit()


def solution_path(conn: sqlite3.Connection, spot_key: str) -> str | None:
    row = conn.execute(
        "SELECT path FROM solution_index WHERE spot_key=?", (spot_key,)).fetchone()
    return row["path"] if row is not None else None


# --------------------------------------------------------------------------- #
# batch_queue (tier-2 solve backlog: miss -> pending; worker drains it)
# --------------------------------------------------------------------------- #
def enqueue_batch(conn: sqlite3.Connection, spot_key: str, hand_id: int,
                  decision_idx: int) -> int:
    cur = conn.execute(
        "INSERT INTO batch_queue(spot_key, hand_id, decision_idx, status)"
        " VALUES (?, ?, ?, 'pending')",
        (spot_key, hand_id, decision_idx),
    )
    conn.commit()
    return int(cur.lastrowid)


def pending_batch(conn: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM batch_queue WHERE status='pending' ORDER BY id")]


def batch_rows(conn: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM batch_queue ORDER BY id")]


def recover_running_batch(conn: sqlite3.Connection) -> int:
    """Reset rows stranded at 'running' back to 'pending'; return the count.

    A drain that dies mid-row (crash, kill, exception before the status is
    finalized) leaves its row at 'running', where no later drain would ever pick
    it up again. Recovery runs at drain start — single-user, single-writer, so a
    'running' row at that moment is by definition abandoned (finding [10]).
    """
    cur = conn.execute(
        "UPDATE batch_queue SET status='pending' WHERE status='running'")
    conn.commit()
    return int(cur.rowcount)


def set_batch_status(conn: sqlite3.Connection, row_id: int, status: str) -> None:
    if status not in ("pending", "running", "done", "failed"):
        raise ValueError(f"bad batch status {status!r}")
    conn.execute("UPDATE batch_queue SET status=? WHERE id=?", (status, row_id))
    conn.commit()
