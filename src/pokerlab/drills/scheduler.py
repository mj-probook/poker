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
# Monopoly bounds (round-3 ruling on [A4]/[P2']). TWO independent limits,
# because the first one alone provably does not do the job:
#
#   RUN  — after this many drills in a row on one category, move on.
#   SHARE— a category over this fraction of the trailing window is de-ranked
#          below everything else until it drops back under.
#
# The run cap alone bounds the RUN and not the SHARE. Capping runs at 3 turned
# a 40/40 monopoly into a perfect `.LLL.LLL.LLL...` duty cycle: the invariant
# held on every single serve while the user still spent 75% of the evening on
# one category and 21 of 32 categories went untouched. A bound that its own
# failure mode satisfies is not a bound, so the share is limited directly.
#
# 40% is a product judgment, not arithmetic: 16 of 40 drills on your worst leak
# is a strong plurality — targeted practice is the entire point of resurfacing —
# while the other 24 slots still interleave the syllabus. 75% is a stuck queue.
#
# Slot arithmetic for the run cap, with its assumptions made explicit so future
# tuning is informed rather than surprised. For a session of S drills over a
# vocabulary of D categories, one category persistently ranking worst, the run
# cap alone admits at most
#
#     distinct categories reached = min(D, 1 + floor(S / (RUN_CAP + 1)))
#
# i.e. the ceiling is f(cap, session length, vocabulary) — NOT a constant. At
# S=40, D=12: cap=1 -> 12/12, cap=2 -> 12/12 (14 slots, vocabulary-limited),
# cap=3 -> 11/12, cap=4 -> 9/12. That ceiling is why an earlier "12/12 coverage"
# pass condition was unsatisfiable at cap=3, and why [E49] tripling the
# vocabulary to D=32 changed the answer again without anything in the scheduler
# moving. cap=1 was rejected: it round-robins away the resurfacing this exists
# to do.
CONSECUTIVE_SERVE_CAP = 2
SHARE_WINDOW = 40        # trailing serves the share is measured over
SHARE_CAP = 0.40         # max fraction of that window one category may hold

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


def _capped_category(conn: sqlite3.Connection) -> str | None:
    """The category that has just been served CONSECUTIVE_SERVE_CAP times running.

    A lapse is due immediately (SM-2, round-2 finding [E51]) and a persistently
    failed category keeps the worst error rate, so the two together let one
    category take every drill in a session — measured 40/40 for three of five
    days against an agent whose leak never resolves, which is exactly the user
    the product exists for (round-3 finding [A4]/[P2']). The lapse semantics are
    right and stay; this bounds the monopoly they permit.

    Derived from `drill_attempts` rather than held in memory so it survives a
    restart and cannot drift from what was actually served.
    """
    rows = conn.execute(
        "SELECT leak_key FROM drill_attempts ORDER BY id DESC LIMIT ?",
        (CONSECUTIVE_SERVE_CAP,)).fetchall()
    if len(rows) < CONSECUTIVE_SERVE_CAP:
        return None
    head = rows[0]["leak_key"]
    return head if all(r["leak_key"] == head for r in rows) else None


def _oversubscribed(conn: sqlite3.Connection) -> set[str]:
    """Categories holding more than SHARE_CAP of the trailing serve window.

    The share is measured over a fixed-width window (SHARE_WINDOW), not over
    "however many rows exist yet": a partial window makes the denominator tiny
    early on, so the second serve of a fresh session reads as 100% and the
    bound fires on noise. With a fixed denominator a category has to actually
    accumulate SHARE_CAP*SHARE_WINDOW serves before it is de-ranked.

    Like `_capped_category` this reads `drill_attempts` — what was actually
    SERVED — so it survives a restart and cannot drift from reality. The window
    deliberately spans sessions: a leak that ate yesterday evening should start
    today already de-ranked, not with a clean slate.
    """
    rows = conn.execute(
        "SELECT leak_key FROM drill_attempts ORDER BY id DESC LIMIT ?",
        (SHARE_WINDOW,)).fetchall()
    if not rows:
        return set()
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["leak_key"]] = counts.get(r["leak_key"], 0) + 1
    # `>=`, not `>`: de-rank the category that has REACHED the allowance, so it
    # never exceeds it. `n > limit` lets a category sit at limit+1 (17/40 =
    # 42.5% against a 40% bound) — the bound would be breached by exactly the
    # serve that triggers it.
    limit = SHARE_CAP * SHARE_WINDOW
    return {k for k, n in counts.items() if n >= limit}


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

    # An over-subscribed category sorts BELOW every other candidate rather than
    # being removed: de-ranking composes with the priority order instead of
    # replacing it, and it degrades gracefully — when the over-subscribed
    # category is the only thing left it is still served, so the bound can
    # never leave the user without a drill. False < True, so this key sinks it.
    over = _oversubscribed(conn)

    def rank(s: SRState) -> tuple[bool, float, float, datetime]:
        p = prio.get(s.leak_key, _NO_PRIORITY)
        return (s.leak_key in over, -p["error_rate"],
                -p["ev_loss_per_100"], _due_dt(s))

    blocked = _capped_category(conn)
    due_keys = [s.leak_key for s in sorted((s for s in states if _is_due(s, now)),
                                           key=rank)]
    soon_keys = [s.leak_key for s in next_due(states)]
    seen = {s.leak_key for s in states}
    unseen = [c for c in known if c not in seen] if known is not None else []

    def first(pool: list[str], *, drop_over: bool) -> str | None:
        for k in pool:
            if k == blocked or (drop_over and k in over):
                continue
            return k
        return None

    # Both bounds are applied by SKIPPING, at every rung — not by sorting.
    #
    # Sorting an over-subscribed category last is inert exactly when it matters
    # most: a persistent lapse is due-now while every correctly-answered
    # category is stamped a day out, so `due` frequently holds that ONE key and
    # ordering a single-element list changes nothing. The same inertness hit
    # the run cap first (it was excluded from the due loop but not from the
    # fallbacks, and the fallbacks are where the monopoly actually landed).
    # Skipping, with a relaxation ladder underneath, is what makes either bound
    # real.
    #
    # The ladder degrades one constraint at a time, so the bounds can never
    # leave the user with no drill: prefer a category under BOTH bounds (due,
    # then never-seen, then soonest-due even if slightly early), and only then
    # readmit an over-subscribed one. Serving a category a few minutes early is
    # the price of the bounds, and it is the right price — the alternative
    # measured 40/40 on a single category.
    for pool, drop_over in ((due_keys, True), (unseen, True), (soon_keys, True),
                            (due_keys, False), (soon_keys, False)):
        pick = first(pool, drop_over=drop_over)
        if pick is not None:
            return pick
    # Only the run-capped category exists at all: yield rather than dead-end.
    # The bounds limit a monopoly; they do not refuse to serve.
    return due_keys[0] if due_keys else (soon_keys[0] if soon_keys else None)
