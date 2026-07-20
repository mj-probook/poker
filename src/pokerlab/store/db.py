"""SQLite wrapper for the checked-in schema (impl doc §2; plan §3 storage).

This module owns exactly two things: applying `store/schema.sql` to a fresh
connection, and the typed insert/query helpers for every app table:

  * `drill_attempts`, `sr_state` — Slice E's drill loop
  * `gradings` — so the tier-3 honesty trigger is reachable from Python
  * `imported_hands` — HH import, including UNIQUE(site, hand_uid) dedup
  * `solution_index` — the tier-2 solve library (`index_solution` is the ONE
    writer; a second copy in `solver.solve_io` was deleted in wave-2 [Q1])
  * `batch_queue` — the tier-2 backlog, including `recover_running_batch`
    (un-strand rows from a crashed drain) and `retry_failed_batch`

Every *derived* metric — leak rankings, accuracy trends — is a query in
`store/views.py`, never a table here (plan §3).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

# Must match the CHECK on batch_queue.status in schema.sql. Kept as one named
# constant because the previous Python-side literal silently drifted from the
# schema the moment a status was added: the CHECK accepted the new value and
# this guard rejected it.
#
#   pending/running  in flight
#   done             graded
#   failed           a bug failed the row -- fix it, then retry_failed_batch
#   unsolvable       structurally outside the solver -- needs a solver upgrade
#   mismatched       the queue no longer describes the hand (hh/persist [B2])
BATCH_STATUSES = frozenset(
    {"pending", "running", "done", "failed", "unsolvable", "mismatched"})


def schema_sql() -> str:
    return SCHEMA_PATH.read_text()


def connect(path: str | Path = ":memory:", *, check_same_thread: bool = True
            ) -> sqlite3.Connection:
    """Open a connection with the schema applied and dict-like row access."""
    conn = sqlite3.connect(str(path), check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    # SQLite ignores REFERENCES clauses unless this is ON, per connection. The
    # schema has declared `gradings.hand_id REFERENCES imported_hands(id)` since
    # Slice E, so it read as a guarantee while enforcing nothing: a grading could
    # point at a hand that was never imported (wave-3 [B2]). Set before the
    # schema runs so every connection is enforcing from its first statement.
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(schema_sql())
    _ensure_current_schema(conn)
    conn.commit()
    return conn


def _ensure_current_schema(conn: sqlite3.Connection) -> None:
    """Bring a PRE-EXISTING database up to the current schema, idempotently.

    `CREATE TABLE IF NOT EXISTS` does nothing to a table that already exists, so
    a `pokerlab.db` created before a schema change silently keeps the old shape.
    For a nullable column that just degrades a feature. For the wave-3 [B3]
    uniqueness rule it is worse: the protection against double-grading a
    decision would be absent on exactly the databases that already hold real
    data.

    No migration framework — this is a single-user local tool and versioned
    migrations would be speculative infrastructure. Two idempotent statements.
    """
    # A legacy DB predating the table gets it from `executescript` above
    # (CREATE TABLE IF NOT EXISTS does create a MISSING table — it is only a
    # no-op for one that already exists), so failed_hands needs no ALTER here.
    # Asserted by test rather than assumed, since that asymmetry is exactly what
    # made the column case need this function at all.
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(gradings)")}
    for name, decl in (("frequency", "REAL"), ("flags", "TEXT")):
        if name not in cols:
            conn.execute(f"ALTER TABLE gradings ADD COLUMN {name} {decl}")

    # SQLite cannot ALTER a UNIQUE constraint into an existing table, but a
    # unique index is equivalent enforcement.
    try:
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS gradings_hand_decision_uq"
                     " ON gradings(hand_id, decision_idx)")
    except sqlite3.IntegrityError as exc:
        # Creation fails only if the data ALREADY violates it. That is not a
        # migration problem to skip past — it means this database contains
        # duplicate gradings for a decision, which double-count into every leak
        # statistic downstream. Fail loudly and name it.
        raise sqlite3.IntegrityError(
            "this database already contains DUPLICATE gradings for the same "
            "(hand_id, decision_idx), so the wave-3 [B3] uniqueness rule cannot "
            "be applied. Those rows double-count into every leak statistic. "
            "Inspect with:\n"
            "  SELECT hand_id, decision_idx, COUNT(*) c FROM gradings\n"
            "  GROUP BY hand_id, decision_idx HAVING c > 1;"
        ) from exc

    # failed_hands: recording a failure is IDEMPOTENT per (site, raw).
    #
    # Deliberately COLLAPSED rather than refused, which is the opposite of the
    # gradings case above, and the difference is what the duplicates mean.
    # Duplicate gradings are distinct rows carrying distinct data that
    # double-count into leak statistics — silently picking one would destroy
    # information, so the only honest move is to stop and name it. Duplicate
    # failure rows are the SAME hand recorded twice by two import runs; keeping
    # the newest is exactly the semantics being adopted, so there is nothing to
    # lose and nothing to ask the user about.
    conn.execute(
        "DELETE FROM failed_hands WHERE id NOT IN ("
        "  SELECT MAX(id) FROM failed_hands GROUP BY site, raw)")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS failed_hands_site_raw_uq"
                 " ON failed_hands(site, raw)")


# --------------------------------------------------------------------------- #
# drill_attempts
# --------------------------------------------------------------------------- #
def insert_drill_attempt(conn: sqlite3.Connection, leak_key: str, kind: str,
                         chosen: str, correct: bool, ev_loss: float,
                         ts: str) -> int:
    cur = conn.execute(
        "INSERT INTO drill_attempts(leak_key, kind, chosen, correct, ev_loss, ts)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (leak_key, kind, chosen, int(bool(correct)), float(ev_loss), ts),
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


def seed_sr_state(conn: sqlite3.Connection, leak_key: str, easiness: float,
                  interval_days: float, reps: int, due: str, *,
                  commit: bool = True) -> None:
    """Make a category drillable at `due` WITHOUT resetting its SM-2 progress.

    `upsert_sr_state` records a *review* and overwrites the whole row. This
    writes the row only if the category is new, and otherwise touches the due
    date alone: an HH import re-opens a leak for drilling, it must not un-learn
    the easiness/interval the drill loop already earned for that category
    (wave-3 [P1']).
    """
    conn.execute(
        "INSERT INTO sr_state(leak_key, easiness, interval_days, reps, due)"
        " VALUES (?, ?, ?, ?, ?)"
        " ON CONFLICT(leak_key) DO UPDATE SET due=excluded.due",
        (leak_key, float(easiness), float(interval_days), int(reps), due),
    )
    if commit:
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
                   leak_key: str, graded_at: str, *, commit: bool = True,
                   frequency: float | None = None,
                   flags: Iterable[str] = ()) -> int:
    """Insert one graded decision.

    `frequency`/`flags` are the reference's read on the hero's action — for
    tier 3 they are the ONLY output it may carry (plan §5.3), so they are
    optional here but never optional in meaning. Flags are stored comma-joined.
    """
    cur = conn.execute(
        "INSERT INTO gradings(hand_id, decision_idx, tier, chosen, best,"
        " ev_loss, leak_key, graded_at, frequency, flags)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (hand_id, decision_idx, tier, chosen, best,
         None if ev_loss is None else float(ev_loss), leak_key, graded_at,
         None if frequency is None else float(frequency), ",".join(flags)),
    )
    if commit:
        conn.commit()
    return int(cur.lastrowid)


def gradings(conn: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT * FROM gradings ORDER BY id")]


# --------------------------------------------------------------------------- #
# imported_hands (Slice F: raw HH + parsed summary, so the batch worker can
# re-derive a decision from storage when a solve finally lands)
# --------------------------------------------------------------------------- #
def insert_failed_hand(conn: sqlite3.Connection, site: str, raw: str,
                       reason: str, imported_at: str,
                       hand_uid: str | None = None, *,
                       commit: bool = True) -> int:
    """Record a hand that could not be parsed or replayed; return its id.

    Kept OUT of `imported_hands` on purpose: a failed hand must not claim its
    (site, hand_uid), or dedup would make the failure permanent and the hand
    could never be re-imported after a parser fix (wave-2 [E16]/[E17]).

    IDEMPOTENT per (site, raw): re-recording a hand that is still broken
    refreshes the existing row rather than adding one. The nightly cron
    re-imports the same file every night, so stacking would have made one
    unchanged broken hand read as a worsening problem in the report — a count
    that grows without anything getting worse (round-4, found by the [R2']
    mutation gate). The LATEST attempt wins on every field: the table answers
    "what is broken now", so a reason that changes under a parser tweak should
    refresh rather than leave the first-ever message frozen in place.
    """
    cur = conn.execute(
        "INSERT INTO failed_hands(site, hand_uid, raw, reason, imported_at)"
        " VALUES (?, ?, ?, ?, ?)"
        " ON CONFLICT(site, raw) DO UPDATE SET"
        "   hand_uid=excluded.hand_uid, reason=excluded.reason,"
        "   imported_at=excluded.imported_at",
        (site, hand_uid or None, raw, reason, imported_at),
    )
    if commit:
        conn.commit()
    return int(cur.lastrowid)


def clear_failed_hand(conn: sqlite3.Connection, site: str, raw: str, *,
                      commit: bool = True) -> int:
    """Drop any recorded failure for this exact hand text; return rows removed.

    Called when the same hand imports SUCCESSFULLY, which is what happens once
    the parser bug is fixed and the file is re-imported. Without it the failure
    count would keep reporting a hand that now grades fine — the table would
    answer "what is broken" with "what was ever broken", and the summary it
    exists to feed would be permanently wrong.

    KNOWN BOUNDARY, [R2'] (documented, not fixed — deliberately):
    a row written BEFORE 1d80de9 for a MULTI-hand file does not clear. It
    recorded "the whole file failed", which after splitting is no longer a
    representable outcome: no chunk equals the file, so no re-import can match
    it. Single-hand legacy rows DO clear as of 7c5c276, which made the chunker
    byte-faithful — a single-hand file chunks to itself exactly, trailing
    newline and CRLF included. Ruled a boundary rather than a migration because
    zero such databases exist (DEFAULT_DB is absent, and the window that could
    have produced one spans a few hours of the same day); writing migration
    code for zero rows is the speculative-infrastructure rule exactly. If such
    a database ever surfaces, the remedy is manual deletion of the stale row.
    """
    cur = conn.execute(
        "DELETE FROM failed_hands WHERE site=? AND raw=?", (site, raw))
    if commit:
        conn.commit()
    return int(cur.rowcount)


def failed_hands(conn: sqlite3.Connection,
                 imported_at: str | None = None) -> list[dict]:
    """Recorded failures, newest first; optionally one import batch."""
    if imported_at is None:
        rows = conn.execute("SELECT * FROM failed_hands ORDER BY id DESC")
    else:
        rows = conn.execute(
            "SELECT * FROM failed_hands WHERE imported_at=? ORDER BY id DESC",
            (imported_at,))
    return [dict(r) for r in rows]


def insert_imported_hand(conn: sqlite3.Connection, site: str, raw: str,
                         parsed_json: str, imported_at: str,
                         hand_uid: str | None = None, *,
                         commit: bool = True) -> int | None:
    """Insert a hand; return its id, or None if (site, hand_uid) already exists.

    ``commit=False`` lets a caller group many writes into one transaction (see
    `hh.persist.persist_session`, round-1 finding [11]).
    """
    cur = conn.execute(
        "INSERT OR IGNORE INTO imported_hands"
        "(site, raw, parsed_json, imported_at, hand_uid)"
        " VALUES (?, ?, ?, ?, ?)",
        (site, raw, parsed_json, imported_at, hand_uid or None),
    )
    if commit:
        conn.commit()
    return None if cur.rowcount == 0 else int(cur.lastrowid)


def get_imported_hand(conn: sqlite3.Connection, hand_id: int) -> dict | None:
    row = conn.execute(
        "SELECT * FROM imported_hands WHERE id=?", (hand_id,)).fetchone()
    return dict(row) if row is not None else None


# --------------------------------------------------------------------------- #
# solution_index (tier-2 library: a spot_key hit means we can grade exactly)
# --------------------------------------------------------------------------- #
def index_solution(conn: sqlite3.Connection, spot_key: str, path: str | Path,
                   solver_version: str, exploitability: float) -> None:
    """Insert/replace a ``solution_index`` row (upsert on ``spot_key``).

    Accepts a ``Path`` because ``solve_io.write_solve`` returns one.
    """
    conn.execute(
        "INSERT INTO solution_index(spot_key, path, solver_version, exploitability)"
        " VALUES (?, ?, ?, ?)"
        " ON CONFLICT(spot_key) DO UPDATE SET path=excluded.path,"
        "   solver_version=excluded.solver_version,"
        "   exploitability=excluded.exploitability",
        (spot_key, str(path), solver_version, float(exploitability)),
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
                  decision_idx: int, *, commit: bool = True) -> int:
    cur = conn.execute(
        "INSERT INTO batch_queue(spot_key, hand_id, decision_idx, status)"
        " VALUES (?, ?, ?, 'pending')",
        (spot_key, hand_id, decision_idx),
    )
    if commit:
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


TERMINAL_STATUSES = ("failed", "unsolvable", "mismatched")


def retry_failed_batch(conn: sqlite3.Connection,
                       statuses: tuple[str, ...] = ("failed",)) -> int:
    """Reopen terminal rows of the given kinds for another drain; return count.

    Defaults to 'failed' — the ONLY class where retrying is itself the remedy.
    The three terminal causes exist as separate states precisely because they
    need different actions, and reopening them together would collapse that
    distinction again one layer up:

      * 'failed'      a bug failed the row -> fix the bug, then retry. Retrying
                      is the remedy, so this is the default.
      * 'unsolvable'  structurally outside what the solver models -> retry only
                      after a SOLVER UPGRADE widens the gate; before that it
                      re-fails by construction.
      * 'mismatched'  the queue row no longer describes the hand -> the remedy
                      is RE-IMPORT. A retry alone re-derives the same wrong spot
                      and re-fails forever, so blanket-reopening this class sends
                      the operator round a loop (w3-product-builder's evidence).

    Pass ``statuses`` explicitly to reopen the others once their real remedy has
    been applied. Never called automatically (wave-2 [E15]).
    """
    bad = set(statuses) - set(TERMINAL_STATUSES)
    if bad:
        raise ValueError(
            f"not terminal statuses: {sorted(bad)} "
            f"(expected a subset of {list(TERMINAL_STATUSES)})")
    marks = ",".join("?" * len(statuses))
    cur = conn.execute(
        f"UPDATE batch_queue SET status='pending' WHERE status IN ({marks})",
        tuple(statuses))
    conn.commit()
    return int(cur.rowcount)


def set_batch_status(conn: sqlite3.Connection, row_id: int, status: str, *,
                     commit: bool = True) -> None:
    """Set a row's status. ``commit=False`` lets a caller bind the status to the
    write it describes in one transaction (see `hh.persist.drain_batch_queue`)."""
    if status not in BATCH_STATUSES:
        raise ValueError(
            f"bad batch status {status!r} (expected one of {sorted(BATCH_STATUSES)})")
    conn.execute("UPDATE batch_queue SET status=? WHERE id=?", (status, row_id))
    if commit:
        conn.commit()
