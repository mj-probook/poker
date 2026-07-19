"""Slice E — SM-2 spaced repetition scheduler (plan §5.3).

Classic SM-2 due-date math plus the resurfacing policy: worst leak category
first among those currently due. All time is injected (fixed `now`) so the
schedule is deterministic.
"""

from datetime import datetime, timedelta

import pytest

from pokerlab.drills import scheduler as sch
from pokerlab.store import db

NOW = datetime(2026, 7, 19, 12, 0, 0)


def _due_after(state, days):
    return (NOW + timedelta(days=days)).isoformat() == state.due


def test_first_correct_review_schedules_one_day():
    st = sch.review(sch.initial_state("SBjam|preflop|jam|10", NOW), correct=True, now=NOW)
    assert st.reps == 1
    assert st.interval_days == 1.0
    assert st.easiness == pytest.approx(2.6)      # 2.5 + 0.1
    assert _due_after(st, 1)


def test_second_correct_review_schedules_six_days():
    st = sch.initial_state("SBjam|preflop|jam|10", NOW)
    st = sch.review(st, True, NOW)                # reps 1
    st = sch.review(st, True, NOW)                # reps 2
    assert st.reps == 2 and st.interval_days == 6.0
    assert _due_after(st, 6)


def test_third_correct_review_scales_by_easiness():
    st = sch.initial_state("SBjam|preflop|jam|10", NOW)
    for _ in range(3):
        st = sch.review(st, True, NOW)
    assert st.reps == 3
    # interval = round(6 * easiness, 4); easiness = 2.5 + 3*0.1 = 2.8
    assert st.easiness == pytest.approx(2.8)
    assert st.interval_days == pytest.approx(round(6 * 2.8, 4))


def test_wrong_answer_resets_reps_and_lowers_easiness():
    st = sch.initial_state("SBjam|preflop|jam|10", NOW)
    for _ in range(3):
        st = sch.review(st, True, NOW)            # build up reps
    before_ef = st.easiness
    st = sch.review(st, False, NOW)               # lapse
    assert st.reps == 0
    assert st.interval_days == 1.0
    assert st.easiness < before_ef
    assert _due_after(st, 1)


def test_easiness_never_below_floor():
    st = sch.initial_state("x", NOW)
    for _ in range(10):
        st = sch.review(st, False, NOW)
    assert st.easiness == pytest.approx(sch.MIN_EASINESS)


def test_next_due_orders_by_due_date():
    states = [
        sch.SRState("late", 2.5, 6.0, 2, (NOW + timedelta(days=6)).isoformat()),
        sch.SRState("soon", 2.5, 1.0, 1, (NOW + timedelta(days=1)).isoformat()),
        sch.SRState("overdue", 2.5, 0.0, 0, (NOW - timedelta(days=1)).isoformat()),
    ]
    order = [s.leak_key for s in sch.next_due(states)]
    assert order == ["overdue", "soon", "late"]


def test_schedule_attempt_persists_state():
    conn = db.connect(":memory:")
    st = sch.schedule_attempt(conn, "SBjam|preflop|jam|10", correct=True, now=NOW)
    row = db.get_sr_state(conn, "SBjam|preflop|jam|10")
    assert row["reps"] == 1 and row["reps"] == st.reps
    # a second attempt updates the same row in place
    sch.schedule_attempt(conn, "SBjam|preflop|jam|10", correct=True, now=NOW)
    assert db.get_sr_state(conn, "SBjam|preflop|jam|10")["reps"] == 2
    assert len(db.all_sr_state(conn)) == 1


def test_select_next_prefers_worst_error_rate_among_due():
    conn = db.connect(":memory:")
    # two categories, both due now; SBjam has the worse error rate.
    for ok in (0, 0, 0, 1):
        db.insert_drill_attempt(conn, "SBjam|preflop|jam|10", "jamfold", "fold",
                                bool(ok), 0.0, "2026-07-19T00:00:00")
    for ok in (1, 1, 1, 0):
        db.insert_drill_attempt(conn, "BBcall|preflop|call|10", "jamfold", "fold",
                                bool(ok), 0.0, "2026-07-19T00:00:00")
    for key in ("SBjam|preflop|jam|10", "BBcall|preflop|call|10"):
        db.upsert_sr_state(conn, key, 2.5, 0.0, 0, NOW.isoformat())  # due now
    assert sch.select_next(conn, NOW) == "SBjam|preflop|jam|10"


def test_select_next_none_when_no_history():
    conn = db.connect(":memory:")
    assert sch.select_next(conn, NOW) is None


def test_select_next_falls_back_to_soonest_when_nothing_due():
    conn = db.connect(":memory:")
    db.upsert_sr_state(conn, "a", 2.5, 6.0, 2, (NOW + timedelta(days=6)).isoformat())
    db.upsert_sr_state(conn, "b", 2.5, 1.0, 1, (NOW + timedelta(days=2)).isoformat())
    assert sch.select_next(conn, NOW) == "b"     # nothing due -> soonest


# --------------------------------------------------------------------------- #
# Round-1 finding [16]: _is_due compared a stored ISO timestamp against `now`
# without normalizing tz-awareness, so a naive stored due vs an aware now (or
# vice versa) raised TypeError instead of answering the question.
# --------------------------------------------------------------------------- #
def test_is_due_handles_mixed_timezone_awareness():
    from datetime import timezone

    from pokerlab.drills.scheduler import SRState, _is_due

    naive_due = SRState("k", 2.5, 0.0, 0, "2026-07-19T00:00:00")
    aware_due = SRState("k", 2.5, 0.0, 0, "2026-07-19T00:00:00+00:00")
    aware_now = datetime(2026, 7, 20, tzinfo=timezone.utc)
    naive_now = datetime(2026, 7, 20)

    # all four combinations must answer, not raise
    assert _is_due(naive_due, aware_now) is True
    assert _is_due(aware_due, naive_now) is True
    assert _is_due(naive_due, naive_now) is True
    assert _is_due(aware_due, aware_now) is True


def test_is_due_is_false_before_the_due_date_across_awareness():
    from datetime import timezone

    from pokerlab.drills.scheduler import SRState, _is_due

    naive_due = SRState("k", 2.5, 0.0, 0, "2026-07-21T00:00:00")
    aware_now = datetime(2026, 7, 20, tzinfo=timezone.utc)
    assert _is_due(naive_due, aware_now) is False
