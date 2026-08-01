"""Multiway drills through the web layer — the tier-3 contract on the wire.

What must be true on the page: no correct/incorrect verdict, no EV number,
ever (CLAUDE.md hard rule) — the response says so in the server's words,
carries the population frequency where a baseline exists ("no baseline" is
stated, never silently zero), and ships the ADVISORY block with its caveat
sentence so the page cannot render solver numbers without the label.
"""

import pytest
from fastapi.testclient import TestClient

from pokerlab.drills.multiway import multiway_drills
from pokerlab.web.app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app(db_path=":memory:", seed=0)
    with TestClient(app) as c:
        yield c


def _river_drill():
    return next(d for d in multiway_drills() if len(d.board) == 5)


def _flop_drill():
    return next(d for d in multiway_drills() if len(d.board) == 3)


def test_multiway_answer_never_claims_correctness_or_ev(client):
    d = _river_drill()
    r = client.post("/api/drill/answer",
                    json={"drill_id": d.drill_id, "action": "bet"})
    assert r.status_code == 200
    body = r.json()
    assert body["correct"] is None
    assert body["ev_loss"] is None
    assert "tier 3" in body["explanation"].lower()
    assert "no ev grading" in body["explanation"].lower()


def test_multiway_answer_carries_advisory_with_caveat(client):
    d = _river_drill()
    # replay guard: same drill twice returns the cached response — vary hand
    r = client.post("/api/drill/answer",
                    json={"drill_id": d.drill_id, "action": "check"})
    body = r.json()
    adv = body["advisory"]
    assert adv is not None
    assert "uniform" in adv["caveat"]
    assert "solve" in adv                    # river: certified HU-collapsed
    assert body["rta"] is None               # no solver READOUT — tier 3


def test_flop_multiway_grades_against_the_population_table(client):
    d = _flop_drill()
    r = client.post("/api/drill/answer",
                    json={"drill_id": d.drill_id, "action": "bet"})
    body = r.json()
    # 6max:BB|flop HAS a baseline row — the frequency is stated
    assert body["frequency_baseline"] == pytest.approx(0.25)
    assert "solve" not in body["advisory"]   # no flop solve exists to show


def test_river_multiway_states_missing_baseline(client):
    d = next(x for x in multiway_drills()
             if len(x.board) == 5 and x.drill_id != _river_drill().drill_id)
    r = client.post("/api/drill/answer",
                    json={"drill_id": d.drill_id, "action": "bet"})
    body = r.json()
    assert body["frequency_baseline"] is None
    assert "no population baseline" in body["explanation"].lower()


def test_multiway_spot_payload_and_filter(client):
    spot = client.get("/api/drill/next?mode=multiway").json()
    assert spot["kind"] == "multiway"
    assert spot["ev_unit"] == ""             # no EV unit exists to advertise
    assert len(spot["seats"]) == 3
    assert spot["legal_actions"] == ["check", "bet"]
    # players=3 alone may serve the 3-max ring tables too (the table-size
    # axis); combined with the mode it must land on a multiway drill
    spot3 = client.get("/api/drill/next?mode=multiway&players=3").json()
    assert spot3["kind"] == "multiway"


def test_store_accepts_and_isolates_null_correct():
    """The schema extension under the drill: correct is nullable (tier 3
    makes no right/wrong claim), and NULL attempts carry no error signal —
    they never rank a category in the leak report and never dilute
    accuracy (SQLite aggregates skip NULLs; the ranking query filters)."""
    from datetime import datetime, timezone

    import pokerlab.store.db as db
    from pokerlab.store.views import (overall_accuracy,
                                      worst_leak_categories)

    conn = db.connect(":memory:")
    ts = datetime.now(timezone.utc).isoformat()
    db.insert_drill_attempt(conn, "mw3.x|flop|root|-", "multiway", "bet",
                            None, None, ts)
    db.insert_drill_attempt(conn, "SBjam|preflop|jam|10", "jamfold", "jam",
                            True, 0.0, ts)
    assert overall_accuracy(conn) == 1.0     # the NULL never counted
    ranked = worst_leak_categories(conn, limit=10)
    assert [r["leak_key"] for r in ranked] == ["SBjam|preflop|jam|10"]
