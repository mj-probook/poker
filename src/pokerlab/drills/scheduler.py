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

import math
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from pokerlab.store import db, views

Q_CORRECT = 5   # clean recall
Q_WRONG = 2     # lapse (< 3 -> schedule resets)
DEFAULT_EASINESS = 2.5
MIN_EASINESS = 1.3
# Ceilings (round-2 finding [E41]). Textbook SM-2 bounds easiness below but not
# above, and lets the interval compound without limit: since a correct answer
# also *raises* the multiplier, the interval grows super-exponentially and
# `now + timedelta(days=interval)` overflows `datetime.max` at ~14 consecutive
# correct reviews — i.e. the app crashed as a reward for mastery. A drill
# syllabus has no use for an interval beyond a year, so both are capped.
MAX_EASINESS = 3.0
MAX_INTERVAL_DAYS = 365.0

# A category no evidence has touched yet ranks as "nothing known", not as
# "known to be fine" — both signals absent read 0 (see `select_next`).
_NO_PRIORITY = {"error_rate": 0.0, "ev_loss_per_100": 0.0}


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
    return min(MAX_EASINESS, max(MIN_EASINESS, ef2))


def _clamp_interval(days: float) -> float:
    """Keep the interval inside [0, MAX_INTERVAL_DAYS]; NaN reads as 0."""
    if not math.isfinite(days):
        return 0.0
    return min(MAX_INTERVAL_DAYS, max(0.0, days))


def _due_iso(now: datetime, interval_days: float) -> str:
    """`now + interval_days` as ISO-8601, total over every input.

    The interval is already clamped, but `now` itself can sit close enough to
    `datetime.max` that the addition still overflows; saturating keeps this
    function total so no scheduling path can raise (round-2 finding [E41]).
    """
    try:
        return (now + timedelta(days=_clamp_interval(interval_days))).isoformat()
    except (OverflowError, OSError, ValueError):
        # Saturate in UTC, not in `now`'s zone (round-3 finding [A3]): a
        # datetime.max carrying a non-UTC offset overflows again the moment
        # anything calls .astimezone(utc) on it, which _due_dt does on every
        # comparison and sort. UTC is the one zone that conversion is a no-op in.
        return datetime.max.replace(tzinfo=timezone.utc).isoformat()


def review(state: SRState, correct: bool, now: datetime) -> SRState:
    """Apply one SM-2 review to `state` (pure).

    A lapse is due again IMMEDIATELY. SM-2 specifies that a quality < 3 response
    restarts the item "from the beginning" and repeats it within the *same*
    session until it is answered well; the reset interval governs the next
    session, not this one. Stamping a lapse `now + 1 day` skipped that
    same-session repeat entirely, which is what let a just-detected leak sit
    undrillable behind never-practised categories (round-2 finding [E51]).
    """
    quality = Q_CORRECT if correct else Q_WRONG
    easiness = _update_easiness(state.easiness, quality)
    if not correct:                       # quality < 3 -> lapse, restart
        reps = 0
        interval = 1.0                    # applies to the NEXT session
        due_in = 0.0                      # ...but drill it again right now
    else:
        reps = state.reps + 1
        if reps == 1:
            interval = 1.0
        elif reps == 2:
            interval = 6.0
        else:
            interval = _clamp_interval(round(state.interval_days * easiness, 4))
        due_in = interval
    return SRState(state.leak_key, easiness, interval, reps, _due_iso(now, due_in))


def next_due(states: list[SRState]) -> list[SRState]:
    """Categories ordered by due date, soonest first."""
    return sorted(states, key=_due_dt)


def _as_utc(ts: datetime) -> datetime:
    """Normalize to aware UTC; naive timestamps are read as UTC.

    Stored due dates and the caller's `now` can differ in tz-awareness (the
    table holds whatever isoformat produced), and comparing a naive to an aware
    datetime raises TypeError rather than answering (round-1 finding [16]).
    """
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)


def _due_dt(state: SRState) -> datetime:
    """The state's due date as aware UTC — the sort/compare key.

    Round-1 finding [16] normalized the *comparison* in `_is_due` but left the
    orderings sorting raw ISO strings, which ranks "…T00:00+00:00" before an
    earlier-sorting-but-later "…T23:00-05:00" (round-2 finding [E44]). An
    unparseable stored value reads as maximally overdue so it gets served and
    re-stamped rather than crashing the whole ordering.

    An OverflowError is caught alongside the parse errors (round-3 finding
    [A3]): a saturated far-future due date stored with a non-UTC offset raises
    from .astimezone() rather than from parsing, and an uncaught raise here
    takes down every ordering, not just the one row.
    """
    try:
        return _as_utc(datetime.fromisoformat(state.due))
    except (TypeError, ValueError, OverflowError):
        return datetime.min.replace(tzinfo=timezone.utc)


def _is_due(state: SRState, now: datetime) -> bool:
    return _due_dt(state) <= _as_utc(now)


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


def select_next(conn: sqlite3.Connection, now: datetime,
                categories: Iterable[str] | None = None) -> str | None:
    """The leak_key to drill next: due first (worst error-rate, then worst
    HH EV-loss), then any never-drilled category, then the soonest-due one.

    Ranking reads `views.category_priority`, which unions the two sources of
    evidence about a category — drill error-rate AND the EV-loss the HH import
    priced from real hands. Reading drill attempts alone made the M4->M2 join
    vacuous: a leak the import had just detected had no drill history, so it
    scored 0 and sat behind whatever the drill loop happened to know about
    (round-3 finding [P1']). Error-rate stays primary — a category being
    answered wrong right now outranks a historical EV price — and EV-loss
    breaks the tie ahead of the due date.

    `categories` is the caller's full category vocabulary — normally every
    `leak_key` the drill generator can emit. Without it this can only ever
    return a key that is already in `sr_state`, and `sr_state` is only written
    by *answering* a drill: the first category answered becomes the only one
    ever served again (round-2 finding [E42]). Passing the population makes
    unseen categories reachable, and confines the result to keys the caller can
    actually serve — a stale `sr_state` row for a category the generator no
    longer emits is skipped rather than silently widening the caller's pool.

    Returns None when there is nothing to schedule at all.
    """
    states = [_state_from_row(r) for r in db.all_sr_state(conn)]
    known = list(dict.fromkeys(categories)) if categories is not None else None
    if known is not None:
        allowed = set(known)
        states = [s for s in states if s.leak_key in allowed]
    if not states and not known:
        return None

    prio = {r["leak_key"]: r for r in views.category_priority(conn)}

    def rank(s: SRState) -> tuple[float, float, datetime]:
        p = prio.get(s.leak_key, _NO_PRIORITY)
        return (-p["error_rate"], -p["ev_loss_per_100"], _due_dt(s))

    due = [s for s in states if _is_due(s, now)]
    if due:
        due.sort(key=rank)
        return due[0].leak_key

    if known is not None:                       # nothing due -> anything unseen
        seen = {s.leak_key for s in states}
        unseen = [c for c in known if c not in seen]
        if unseen:
            return unseen[0]

    return next_due(states)[0].leak_key if states else None
