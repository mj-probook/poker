"""Slice E — SM-2 spaced repetition scheduler (plan §5.3).

Classic SM-2 due-date math plus the resurfacing policy: worst leak category
first among those currently due. All time is injected (fixed `now`) so the
schedule is deterministic.
"""

import math
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
    assert st.easiness < before_ef
    # The reset interval governs the NEXT session...
    assert st.interval_days == 1.0
    # ...but SM-2 repeats a lapsed item within the SAME session, so it is due
    # immediately. This assertion used to read `_due_after(st, 1)`, which is the
    # behaviour that let a just-detected leak sit undrillable for a day behind
    # never-practised categories (round-2 finding [E51]).
    assert _due_after(st, 0)
    assert sch._is_due(st, NOW) is True


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


# --------------------------------------------------------------------------- #
# Round-2 finding [E41]: easiness had a floor but no ceiling and the interval
# compounded unbounded, so `now + timedelta(days=interval)` overflowed
# datetime.max after ~14 consecutive correct reviews -- a 500 on the success
# path. Round-2 [E42]: select_next could only ever return a key already in
# sr_state, so the first category answered was the only one ever served again.
# Round-2 [E44]: the orderings sorted raw ISO strings instead of instants.
# --------------------------------------------------------------------------- #
def test_interval_and_easiness_are_capped():
    st = sch.initial_state("k", NOW)
    for _ in range(40):
        st = sch.review(st, True, NOW)
    assert st.easiness <= sch.MAX_EASINESS
    assert st.interval_days <= sch.MAX_INTERVAL_DAYS


@pytest.mark.parametrize("reps", [14, 100, 10_000])
def test_review_never_overflows_however_long_the_streak(reps):
    """The whole point of [E41]: no streak length may raise."""
    st = sch.initial_state("k", NOW)
    for _ in range(reps):
        st = sch.review(st, True, NOW)          # must not raise OverflowError
    assert datetime.fromisoformat(st.due) >= NOW


def test_due_arithmetic_is_total_even_next_to_datetime_max():
    st = sch.SRState("k", 2.5, 6.0, 5, datetime.max.isoformat())
    out = sch.review(st, True, datetime.max - timedelta(days=1))
    assert datetime.fromisoformat(out.due)      # saturated, not raised


def test_select_next_reaches_categories_never_drilled():
    conn = db.connect(":memory:")
    cats = ["a", "b", "c"]
    # "a" answered and scheduled far out; b and c have never been drilled.
    db.upsert_sr_state(conn, "a", 2.5, 6.0, 2, (NOW + timedelta(days=6)).isoformat())
    assert sch.select_next(conn, NOW, categories=cats) in {"b", "c"}
    # without the vocabulary the scheduler can still only see what it has seen
    assert sch.select_next(conn, NOW) == "a"


def test_select_next_skips_stale_keys_outside_the_vocabulary():
    conn = db.connect(":memory:")
    db.upsert_sr_state(conn, "OBSOLETE|preflop|jam|3", 2.5, 0.0, 0,
                       (NOW - timedelta(days=999)).isoformat())   # maximally overdue
    assert sch.select_next(conn, NOW, categories=["a", "b"]) in {"a", "b"}


def test_next_due_orders_by_instant_not_iso_string():
    """+00:00 sorts before -05:00 lexicographically but is the EARLIER instant."""
    states = [
        sch.SRState("utc", 2.5, 0.0, 0, "2026-07-19T00:00:00+00:00"),   # 00:00Z
        sch.SRState("offset", 2.5, 0.0, 0, "2026-07-18T23:00:00-05:00"),  # 04:00Z
    ]
    assert [s.leak_key for s in sch.next_due(states)] == ["utc", "offset"]
    assert [s.leak_key for s in sch.next_due(list(reversed(states)))] == ["utc", "offset"]


def test_unparseable_due_is_treated_as_overdue_not_a_crash():
    bad = sch.SRState("k", 2.5, 0.0, 0, "not-a-timestamp")
    assert sch._is_due(bad, NOW) is True
    assert sch.next_due([bad])[0].leak_key == "k"


# --------------------------------------------------------------------------- #
# Round-2 finding [E51]: E42 taught select_next about never-drilled categories,
# which exposed a latent SM-2 infidelity -- a lapse was stamped `now + 1 day`,
# so a leak detected in a hand history and registered RIGHT NOW was not due
# until tomorrow and lost to every unseen category. That silently broke the
# plan-§5.3 "resurface the worst categories" promise, which is the M4->M2 seam.
# --------------------------------------------------------------------------- #
def test_a_just_registered_leak_outranks_never_drilled_categories():
    conn = db.connect(":memory:")
    top = "SBjam|preflop|jam|10"
    others = ["BBcall.icm|preflop|call|10", "SBjam|preflop|jam|5", top]
    sch.schedule_attempt(conn, top, correct=False, now=NOW)   # the HH leak
    assert sch.select_next(conn, NOW, categories=others) == top


def test_a_lapse_stays_due_until_it_is_answered_correctly():
    """SM-2 repeats a failed item until quality recovers, then schedules out."""
    conn = db.connect(":memory:")
    key = "SBjam|preflop|jam|10"
    cats = [key, "SBjam|preflop|jam|5"]
    for _ in range(3):                       # keep failing -> keeps resurfacing
        sch.schedule_attempt(conn, key, correct=False, now=NOW)
        assert sch.select_next(conn, NOW, categories=cats) == key
    sch.schedule_attempt(conn, key, correct=True, now=NOW)    # finally correct
    assert sch.select_next(conn, NOW, categories=cats) != key  # yields the floor


def test_due_dt_survives_a_saturated_far_future_stamp():
    """[A3] _due_iso saturates to datetime.max; that value must still be
    orderable. Stored with a non-UTC offset it raises OverflowError from
    .astimezone(), not from parsing -- and an uncaught raise here takes down
    every ordering, not just the one row."""
    from datetime import timezone as _tz

    sat = sch._due_iso(datetime.max - timedelta(days=1), 999.0)
    assert sch._due_dt(sch.SRState("k", 2.5, 1.0, 0, sat))       # no raise
    # an explicitly non-UTC saturated stamp is the hostile case
    hostile = datetime.max.replace(tzinfo=_tz(timedelta(hours=-5))).isoformat()
    st = sch.SRState("k", 2.5, 1.0, 0, hostile)
    assert sch._due_dt(st)                                        # no raise
    assert sch.next_due([st, sch.SRState("n", 2.5, 0.0, 0, NOW.isoformat())])


# --------------------------------------------------------------------------- #
# Round-3 findings [A4]/[P2']: a persistently-failing category monopolized the
# session. A lapse is due immediately ([E51], SM-2 fidelity) and a category
# that keeps being answered wrong keeps the worst error rate, so the worst leak
# re-won the ranking on every single pick -- measured 40/40 drills on one
# category for three of five simulated days, against exactly the user this
# product exists for (the one whose leak does NOT resolve). The lapse semantics
# are correct and stay; `CONSECUTIVE_SERVE_CAP` bounds the monopoly they allow.
# --------------------------------------------------------------------------- #
def _serve_and_answer(conn, cats, now, *, failing: str) -> str:
    """One drill-loop turn: ask the scheduler, then record a PERSISTENT miss.

    Mirrors what web/app.py does per answer -- record the attempt AND review
    the SM-2 state. The cap is derived from `drill_attempts` (what was actually
    served) rather than from `sr_state`, so a test that only scheduled would
    never exercise it.
    """
    key = sch.select_next(conn, now, categories=cats)
    assert key is not None
    correct = key != failing            # the seeded leak is NEVER answered right
    db.insert_drill_attempt(conn, key, "jamfold", "fold", correct, 0.0,
                            now.isoformat())
    sch.schedule_attempt(conn, key, correct, now)
    return key


def test_a_persistent_leak_cannot_monopolize_the_session():
    """The post-fix invariant: no category is served more than the cap in a row.

    Asserted on the CAP, not on a coverage count: coverage is a function of the
    vocabulary size (which the [E49] ante rev tripled, 12 -> 32) and of session
    length, so pinning a coverage number bakes today's syllabus into a
    scheduler test. The cap is the actual contract and is independent of both.
    """
    conn = db.connect()
    cats = [f"cat{i}|preflop|jam|10" for i in range(8)]
    leak = cats[0]
    now = NOW
    served = []
    for _ in range(60):
        served.append(_serve_and_answer(conn, cats, now, failing=leak))
        now += timedelta(minutes=1)

    run = worst = 1
    for prev, cur in zip(served, served[1:]):
        run = run + 1 if cur == prev else 1
        worst = max(worst, run)
    assert worst <= sch.CONSECUTIVE_SERVE_CAP, (
        f"{worst} consecutive serves on one category "
        f"(cap is {sch.CONSECUTIVE_SERVE_CAP}); served={served[:20]}")
    # ...and the cap must not have starved the leak either: it is still the
    # worst category and must still get the largest share of the session.
    assert served.count(leak) == max(served.count(c) for c in cats)


def test_a_persistent_leak_cannot_take_more_than_its_share_of_the_window():
    """The run cap bounds the RUN; this bounds the SHARE. Both are needed.

    Capping runs alone converts a 40/40 monopoly into a `.LLL.LLL...` duty
    cycle: the run invariant holds on every serve while the category still
    takes three-quarters of the session. So this asserts the property the run
    cap cannot express -- over EVERY trailing window, no category exceeds
    SHARE_CAP. Sliding the window (rather than checking the total) is what
    makes it a real bound: a category could sit under the limit overall while
    completely owning one stretch.
    """
    conn = db.connect()
    cats = [f"cat{i}|preflop|jam|10" for i in range(8)]
    now = NOW
    served = []
    for _ in range(150):
        served.append(_serve_and_answer(conn, cats, now, failing=cats[0]))
        now += timedelta(minutes=1)

    w = sch.SHARE_WINDOW
    limit = sch.SHARE_CAP * w
    worst, at = 0, 0
    for i in range(len(served) - w + 1):
        window = served[i:i + w]
        n = max(window.count(c) for c in set(window))
        if n > worst:
            worst, at = n, i
    assert worst <= limit, (
        f"{worst}/{w} serves ({worst / w:.0%}) on one category in the window "
        f"starting at {at}; cap is {limit:.0f}/{w} ({sch.SHARE_CAP:.0%})")


def test_the_cap_still_lets_every_other_category_through():
    """The cap must yield to OTHER categories, not merely stall on the leak.

    Guards the test above from passing via a degenerate scheduler that returns
    None or thrashes between two keys: a bounded monopoly is only a fix if the
    session actually spreads.
    """
    conn = db.connect()
    cats = [f"cat{i}|preflop|jam|10" for i in range(8)]
    now = NOW
    served = []
    for _ in range(60):
        served.append(_serve_and_answer(conn, cats, now, failing=cats[0]))
        now += timedelta(minutes=1)
    assert set(served) == set(cats), f"never served {set(cats) - set(served)}"


# --------------------------------------------------------------------------- #
# The struggling user's regime. The test above deliberately declines to pin a
# coverage NUMBER, and that scoping is correct for its own regime: under a
# mixed policy the achieved count really is a function of vocabulary size and
# session length (measured all-correct: D=6/12/32/64 -> 6/12/32/64), so a bare
# number there would bake today's syllabus into a scheduler test.
#
# It does not follow that no count is pinnable. Under an ALL-WRONG policy the
# achieved count stops responding to either variable and becomes a constant set
# by SHARE_CAP alone (round-4 [8]). That regime is not a corner case — it is the
# user this scheduler exists for.
# --------------------------------------------------------------------------- #
def test_a_struggling_user_still_sees_the_share_bound_worth_of_variety():
    """Everything wrong: the achieved variety is SHARE_CAP's, not the syllabus's.

    When every answer is incorrect, every category lapses due-now and none can
    rank its way out, so the due pool is saturated and ranking cannot spread the
    session. What spreads it is the share bound alone: one category may hold at
    most SHARE_CAP of the trailing window, so it takes ceil(1 / SHARE_CAP)
    categories to fill that window, plus one more circulating through churn.

    Asserted against the DERIVED form rather than the literal 4, which is what
    makes it immune to the objection that sank a coverage count in the test
    above: it encodes no syllabus fact. There is no vocabulary size and no
    session length in `expected` — only the constant that actually governs this
    regime. Retune SHARE_CAP and both sides move together by construction;
    verified at c = 0.5/0.4/0.34/0.25/0.2/0.125 giving 3/4/4/5/6/9, achieved
    matching derived at every point including the non-reciprocal 0.34. So this
    fails when the RELATIONSHIP breaks, which is the thing worth knowing, and
    not merely when someone edits a number.

    D and S are a probe point, not a claim: 4 was measured invariant across
    D=6..64 and S=40..150, so any pair in that box tests the same property.

    The value of pinning it: this constant lived only in a comment
    (`scheduler.py` above CONSECUTIVE_SERVE_CAP), and a measured number in prose
    is one nobody re-runs. Raising RUN_CAP or tripling the vocabulary moves the
    upper bound and changes nothing here — only SHARE_CAP moves the floor a
    struggling user actually experiences, and that is the number a tuner needs
    to see move.
    """
    D, S = 32, 80                       # interior of the measured-invariant box
    expected = math.ceil(1 / sch.SHARE_CAP) + 1
    conn = db.connect()
    cats = [f"cat{i}|preflop|jam|10" for i in range(D)]
    now = NOW
    served = []
    for _ in range(S):
        key = sch.select_next(conn, now, categories=cats)
        assert key is not None, "the scheduler must always have something to serve"
        served.append(key)
        db.insert_drill_attempt(conn, key, "jamfold", "fold", False, 0.0,
                                now.isoformat())
        sch.schedule_attempt(conn, key, False, now)
        now += timedelta(minutes=1)

    assert len(set(served)) == expected, (
        f"a struggling user saw {len(set(served))} distinct categories over {S} "
        f"serves of a {D}-category syllabus; SHARE_CAP={sch.SHARE_CAP} predicts "
        f"{expected}. If SHARE_CAP was just retuned this is the floor moving as "
        f"intended — update nothing, the derived form tracked it. If it was NOT, "
        f"the share bound is no longer what governs this regime.")
