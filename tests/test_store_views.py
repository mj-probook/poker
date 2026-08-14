"""Slice E — derived-metric queries over drill_attempts (impl doc §2; plan §3).

Derived metrics are QUERIES, never tables. These check the two Slice-E views:
worst leak categories by error rate, and the accuracy trend over time.
"""

import pytest

from pokerlab.store import db, views


def _seed(conn):
    # SBjam:10bb — mostly wrong (3/4 errors); BBcall:10bb — mostly right (1/4).
    rows = [
        ("SBjam:10bb", 0), ("SBjam:10bb", 0), ("SBjam:10bb", 0), ("SBjam:10bb", 1),
        ("BBcall:10bb", 1), ("BBcall:10bb", 1), ("BBcall:10bb", 1), ("BBcall:10bb", 0),
    ]
    for i, (sk, ok) in enumerate(rows):
        db.insert_drill_attempt(conn, sk, "jamfold", "fold", bool(ok), 0.0,
                                f"2026-07-19T00:00:{i:02d}")


def test_worst_leak_categories_ranked_by_error_rate():
    conn = db.connect(":memory:")
    _seed(conn)
    worst = views.worst_leak_categories(conn, limit=5)
    assert worst[0]["leak_key"] == "SBjam:10bb"
    assert worst[0]["attempts"] == 4
    assert worst[0]["errors"] == 3
    assert worst[0]["error_rate"] == pytest.approx(0.75)
    # the better category ranks below it
    assert worst[1]["leak_key"] == "BBcall:10bb"
    assert worst[1]["error_rate"] == pytest.approx(0.25)


def test_worst_leak_categories_respects_min_attempts():
    conn = db.connect(":memory:")
    db.insert_drill_attempt(conn, "rare:5bb", "jamfold", "fold", False, 0.0, "t0")
    _seed(conn)
    worst = views.worst_leak_categories(conn, limit=5, min_attempts=4)
    keys = {w["leak_key"] for w in worst}
    assert "rare:5bb" not in keys        # only 1 attempt -> filtered out
    assert "SBjam:10bb" in keys


def test_attempt_history_trend_is_cumulative_accuracy():
    conn = db.connect(":memory:")
    # W, L, W, W -> cumulative accuracy 1.0, 0.5, 0.667, 0.75
    for i, ok in enumerate([1, 0, 1, 1]):
        db.insert_drill_attempt(conn, "SBjam:10bb", "jamfold", "jam",
                                bool(ok), 0.0, f"2026-07-19T00:00:{i:02d}")
    trend = views.attempt_history_trend(conn)
    accs = [round(r["cumulative_accuracy"], 4) for r in trend]
    assert accs == [1.0, 0.5, 0.6667, 0.75]


def test_overall_accuracy():
    conn = db.connect(":memory:")
    _seed(conn)
    assert views.overall_accuracy(conn) == pytest.approx(0.5)   # 4 of 8


def test_overall_accuracy_empty_is_zero():
    conn = db.connect(":memory:")
    assert views.overall_accuracy(conn) == 0.0


# --------------------------------------------------------------------------- #
# Round-3 finding [P1'] (P0): the scheduler ranked categories by drill
# error-rate ALONE, so a leak the HH import had just priced in EV could never
# outrank a category the drill loop happened to know about. `category_priority`
# is the join — derived, so it is a QUERY, never a column (plan §3).
# --------------------------------------------------------------------------- #
def _grade(conn, leak_key, ev_loss, tier=1, idx=0):
    hid = db.insert_imported_hand(conn, "PokerStars", "raw", "{}",
                                  "2026-07-19T00:00:00", f"h{leak_key}{idx}")
    db.insert_grading(conn, hid, idx, tier, "jam", "fold", ev_loss, leak_key,
                      "2026-07-19T00:00:00")


def test_category_priority_joins_hh_ev_loss_with_drill_error_rate():
    conn = db.connect(":memory:")
    _seed(conn)                                  # drill attempts only
    _grade(conn, "SBjam|preflop|jam|10", 1.14)   # HH leak, never drilled
    _grade(conn, "BBcall:10bb", 0.02, idx=1)     # drilled AND HH-graded

    rows = {r["leak_key"]: r for r in views.category_priority(conn)}

    # a category known only to the HH import is present, with a real EV price
    assert rows["SBjam|preflop|jam|10"]["ev_loss_per_100"] == pytest.approx(114.0)
    assert rows["SBjam|preflop|jam|10"]["attempts"] == 0
    # a category known only to the drill loop is present, with no EV price
    assert rows["SBjam:10bb"]["error_rate"] == pytest.approx(0.75)
    assert rows["SBjam:10bb"]["ev_loss_per_100"] == 0.0
    # and one the two sources share carries both signals
    both = rows["BBcall:10bb"]
    assert both["error_rate"] == pytest.approx(0.25)
    assert both["ev_loss_per_100"] == pytest.approx(2.0)


def test_category_priority_excludes_tier_3():
    """Tier 3 has no ev_loss, so it can never be ranked here (plan §5.3)."""
    conn = db.connect(":memory:")
    _grade(conn, "6max:LJ|flop|bet|-", None, tier=3)
    assert views.category_priority(conn) == []


def test_category_priority_never_fabricates_a_rate_for_null_correct():
    """A tier-3 drill attempt carries correct=NULL — no right/wrong claim.

    Night-shift review [2]: the dr CTE AVG'd an all-NULL category to NULL and
    the COALESCE then reported error_rate 0.0 — a fabricated "never wrong"
    claim for the exact tier the NULL exists to protect. Such a category must
    simply be ABSENT: no signal from either source, nothing to rank.
    """
    conn = db.connect(":memory:")
    mw = "mw3.BTNCOBB.dry|flop|root"
    for i in range(3):
        db.insert_drill_attempt(conn, mw, "multiway", "check", None, None,
                                f"2026-08-01T00:00:{i:02d}")
    _seed(conn)                              # graded categories still rank
    rows = {r["leak_key"]: r for r in views.category_priority(conn)}
    assert mw not in rows
    assert rows["SBjam:10bb"]["error_rate"] == pytest.approx(0.75)


# --------------------------------------------------------------------------- #
# Round-3 finding [P6'] (P1): plan §5.4 measures progress by "decision-quality
# trends over large samples", and neither trend was computable. Both are
# derived, so both are queries (plan §3) — no schema was added for either.
# --------------------------------------------------------------------------- #
def test_ev_loss_trend_buckets_by_category_over_time():
    conn = db.connect(":memory:")
    # same category, two months, improving; plus a second category
    for i, (month, ev) in enumerate([("01", 2.0), ("01", 4.0), ("02", 1.0)]):
        _grade(conn, "SBjam|preflop|jam|10", ev, idx=i)
        conn.execute("UPDATE gradings SET graded_at=? WHERE id=(SELECT MAX(id)"
                     " FROM gradings)", (f"2026-{month}-15T12:00:00",))
    conn.commit()

    rows = views.ev_loss_per_100_by_category_over_time(conn, bucket="month")

    jan = next(r for r in rows if r["bucket"] == "2026-01")
    feb = next(r for r in rows if r["bucket"] == "2026-02")
    assert jan["ev_loss_per_100"] == pytest.approx(300.0)   # mean(2,4) * 100
    assert feb["ev_loss_per_100"] == pytest.approx(100.0)   # the leak is closing
    assert jan["decisions"] == 2 and feb["decisions"] == 1
    assert [r["bucket"] for r in rows] == sorted(r["bucket"] for r in rows)


def test_ev_loss_trend_never_includes_tier_3():
    """Tier 3 has no ev_loss; a trend that counted it would be a lie."""
    conn = db.connect(":memory:")
    _grade(conn, "6max:LJ|flop|bet|-", None, tier=3)
    assert views.ev_loss_per_100_by_category_over_time(conn) == []


def test_accuracy_by_kind_within_a_window():
    conn = db.connect(":memory:")
    # jamfold: 1/2 correct recently, plus an old attempt outside the window
    db.insert_drill_attempt(conn, "k", "jamfold", "jam", True, 0.0,
                            "2026-07-18T12:00:00")
    db.insert_drill_attempt(conn, "k", "jamfold", "jam", False, 1.0,
                            "2026-07-18T13:00:00")
    db.insert_drill_attempt(conn, "k", "jamfold", "jam", False, 1.0,
                            "2026-01-01T00:00:00")
    db.insert_drill_attempt(conn, "k", "icm", "jam", True, 0.0,
                            "2026-07-18T14:00:00")
    now = "2026-07-19T00:00:00"

    rows = {r["kind"]: r for r in
            views.accuracy_by_kind(conn, window_days=7, now=now)}

    assert rows["jamfold"]["attempts"] == 2                  # old one excluded
    assert rows["jamfold"]["accuracy"] == pytest.approx(0.5)
    assert rows["icm"]["accuracy"] == pytest.approx(1.0)

    only = views.accuracy_by_kind(conn, kind="icm", window_days=7, now=now)
    assert [r["kind"] for r in only] == ["icm"]


def test_accuracy_by_kind_reports_an_empty_window_as_no_rows():
    """No attempts in the window is zero DATA, never zero accuracy."""
    conn = db.connect(":memory:")
    db.insert_drill_attempt(conn, "k", "jamfold", "jam", False, 1.0,
                            "2026-01-01T00:00:00")
    assert views.accuracy_by_kind(conn, window_days=7,
                                  now="2026-07-19T00:00:00") == []


def test_accuracy_by_kind_keeps_the_multiway_kind_honest():
    """Tier-3 attempts (correct=NULL) count as PRACTICE VOLUME but make no
    accuracy claim: the row stays (attempts real), accuracy is None, and it
    sorts BELOW every graded kind — a no-claim row must never outrank a
    genuinely bad one in a worst-first list (night-shift review [2])."""
    conn = db.connect(":memory:")
    for i in range(3):
        db.insert_drill_attempt(conn, "mw", "multiway", "check", None, None,
                                f"2026-07-18T12:00:{i:02d}")
    db.insert_drill_attempt(conn, "k", "jamfold", "jam", False, 1.0,
                            "2026-07-18T13:00:00")
    rows = views.accuracy_by_kind(conn, window_days=7,
                                  now="2026-07-19T00:00:00")
    assert [r["kind"] for r in rows] == ["jamfold", "multiway"]
    mw = rows[1]
    assert mw["attempts"] == 3
    assert mw["accuracy"] is None and mw["avg_ev_loss"] is None
