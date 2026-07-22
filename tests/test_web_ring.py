"""Ring drills through the web layer: seats as server facts, end-to-end grade.

Seat attribution is a formation fact for HU and 9-max ring drills (action
order + who jammed + who must have folded for the hero to be facing this
decision), so the payload states every seat's state and the page draws it.
ICM multiway stays unattributed — its payload has never claimed to know which
stack sits where, and a seats array would fabricate exactly that.
"""

import pytest
from fastapi.testclient import TestClient

from pokerlab.drills import generator as gen
from pokerlab.web.app import _spot_json, create_app


@pytest.fixture(scope="module")
def ring():
    return gen.ring_drills()


def _states(spot):
    return [(s["pos"], s["state"]) for s in spot["seats"]]


def test_ring_jam_spot_carries_every_seat_state(ring):
    d = next(x for x in ring if x.position == "CO" and not x.versus)
    spot = _spot_json(d)
    assert _states(spot) == [
        ("UTG", "folded"), ("UTG1", "folded"), ("UTG2", "folded"),
        ("LJ", "folded"), ("HJ", "folded"), ("CO", "hero"),
        ("BTN", "live"), ("SB", "live"), ("BB", "live")]
    assert spot["action_line"] == "Folded to you — action on you"
    assert spot["facing_allin"] is False


def test_ring_defense_spot_marks_jammer_and_intermediate_folds(ring):
    d = next(x for x in ring if x.position == "BB" and x.versus == "CO")
    spot = _spot_json(d)
    assert _states(spot) == [
        ("UTG", "folded"), ("UTG1", "folded"), ("UTG2", "folded"),
        ("LJ", "folded"), ("HJ", "folded"), ("CO", "all-in"),
        ("BTN", "folded"), ("SB", "folded"), ("BB", "hero")]
    assert "CO is all-in" in spot["action_line"]
    assert spot["facing_allin"] is True


def test_utg_jam_is_action_on_you_not_folded_to_you(ring):
    # nobody is before UTG, so "folded to you" would be a false fact
    d = next(x for x in ring if x.position == "UTG" and not x.versus)
    assert _spot_json(d)["action_line"] == "Action on you"


def test_hu_spot_now_carries_seats_too():
    d = next(x for x in gen.jamfold_drills(depths=(10,))
             if x.position == "BB")
    spot = _spot_json(d)
    assert _states(spot) == [("SB", "all-in"), ("BB", "hero")]


def test_icm_spot_keeps_seats_unattributed():
    d = next(x for x in gen.icm_drills() if x.position == "BB")
    assert _spot_json(d)["seats"] is None


def test_fresh_app_interleaves_kinds_from_the_first_spots(tmp_path):
    """New content must be discoverable IMMEDIATELY, not after the legacy
    queue drains. The unseen rung serves the caller's category order, and
    population order put all 630 ring categories behind every HU/ICM one —
    the user merged the feature, clicked through spots, and saw nothing new
    (2026-07-22 report). Round-robin by kind makes the first three fresh
    spots span all three kinds."""
    app = create_app(db_path=str(tmp_path / "fresh.db"), seed=0)
    kinds = []
    with TestClient(app) as client:
        for _ in range(3):
            spot = client.get("/api/drill/next").json()
            kinds.append(spot["kind"])
            best = max(spot["legal_actions"], key=lambda a: a)  # any answer
            client.post("/api/drill/answer", json={
                "drill_id": spot["drill_id"], "action": best})
    assert set(kinds) == {"jamfold", "icm", "ring"}


def test_nothing_is_browser_cached(tmp_path):
    """Local single-user tool: a cached page is only ever a stale page. Two
    user reports ("no visual card rendering", "I refreshed and still don't
    see it") were this exact failure — the fix is server policy, not user
    keyboard discipline."""
    app = create_app(db_path=str(tmp_path / "cache.db"), seed=0)
    with TestClient(app) as client:
        for path in ("/", "/static/app.js", "/api/drill/next"):
            assert client.get(path).headers.get("cache-control") == "no-store"


def test_ring_answer_grades_end_to_end(tmp_path):
    app = create_app(db_path=str(tmp_path / "ring.db"), seed=0)
    with TestClient(app) as client:
        r = client.post("/api/drill/answer", json={
            "drill_id": "BBcall.vCO|preflop|call|10:AA", "action": "call"})
        assert r.status_code == 200
        body = r.json()
        assert body["correct"] is True         # AA always calls
        assert body["ev_unit"] == "bb"
        assert set(a["action"] for a in body["rta"]["actions"]) == {
            "call", "fold"}
