"""SM-2 spaced repetition over leak categories (Slice E; plan §5.3).

Classic SM-2: each attempt on a `leak_key` updates its easiness factor,
repetition count, and inter-repetition interval, and stamps the next due date.
Quality is binary here — a correct drill answer is a high-quality recall (5), an
incorrect one a lapse (2) that resets the repetition schedule. `sr_state` is the
persisted table; the *ordering* of what to resurface next is a query that puts
the worst-error-rate category first among everything currently due (plan §5.3:
"resurfaces the worst categories as drills").
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from pokerlab.store import db, views

Q_CORRECT = 5   # clean recall
Q_WRONG = 2     # lapse (< 3 -> schedule resets)
DEFAULT_EASINESS = 2.5
MIN_EASINESS = 1.3


@dataclass(frozen=True)
class SRState:
    leak_key: str
    easiness: float
    interval_days: float
    reps: int
    due: str        # ISO-8601 timestamp


def initial_state(leak_key: str, now: datetime) -> SRState:
    """A never-drilled category: default easiness, due immediately."""
    return SRState(leak_key, DEFAULT_EASINESS, 0.0, 0, now.isoformat())


def _update_easiness(ef: float, quality: int) -> float:
    ef2 = ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    return max(MIN_EASINESS, ef2)


def review(state: SRState, correct: bool, now: datetime) -> SRState:
    """Apply one SM-2 review to `state` (pure)."""
    quality = Q_CORRECT if correct else Q_WRONG
    easiness = _update_easiness(state.easiness, quality)
    if not correct:                       # quality < 3 -> lapse, restart
        reps = 0
        interval = 1.0
    else:
        reps = state.reps + 1
        if reps == 1:
            interval = 1.0
        elif reps == 2:
            interval = 6.0
        else:
            interval = round(state.interval_days * easiness, 4)
    due = (now + timedelta(days=interval)).isoformat()
    return SRState(state.leak_key, easiness, interval, reps, due)


def next_due(states: list[SRState]) -> list[SRState]:
    """Categories ordered by due date, soonest first."""
    return sorted(states, key=lambda s: s.due)


def _as_utc(ts: datetime) -> datetime:
    """Normalize to aware UTC; naive timestamps are read as UTC.

    Stored due dates and the caller's `now` can differ in tz-awareness (the
    table holds whatever isoformat produced), and comparing a naive to an aware
    datetime raises TypeError rather than answering (round-1 finding [16]).
    """
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)


def _is_due(state: SRState, now: datetime) -> bool:
    return _as_utc(datetime.fromisoformat(state.due)) <= _as_utc(now)


# --------------------------------------------------------------------------- #
# Persistence-backed helpers.
# --------------------------------------------------------------------------- #
def _state_from_row(row: dict) -> SRState:
    return SRState(row["leak_key"], row["easiness"], row["interval_days"],
                   row["reps"], row["due"])


def schedule_attempt(conn: sqlite3.Connection, leak_key: str, correct: bool,
                     now: datetime) -> SRState:
    """Load-or-create the category's SM-2 state, review it, persist it."""
    row = db.get_sr_state(conn, leak_key)
    state = _state_from_row(row) if row else initial_state(leak_key, now)
    updated = review(state, correct, now)
    db.upsert_sr_state(conn, updated.leak_key, updated.easiness,
                       updated.interval_days, updated.reps, updated.due)
    return updated


def select_next(conn: sqlite3.Connection, now: datetime) -> str | None:
    """The leak_key to drill next: worst error-rate among those due, else the
    soonest-due category. None when nothing has ever been scheduled."""
    states = [_state_from_row(r) for r in db.all_sr_state(conn)]
    if not states:
        return None
    err = {r["spot_key"]: r["error_rate"]
           for r in views.worst_leak_categories(conn, limit=10_000)}
    due = [s for s in states if _is_due(s, now)]
    if due:
        due.sort(key=lambda s: (-err.get(s.leak_key, 0.0), s.due))
        return due[0].leak_key
    return next_due(states)[0].leak_key
