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
