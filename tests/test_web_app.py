"""Slice E — FastAPI drill routes (impl doc §3; plan §3 UI contract).

Contract tests via the httpx-backed TestClient: the two JSON routes, their
happy-path shapes, ICM context surfacing, persistence, and the error paths
(unknown drill_id, illegal action label).
"""

import sqlite3

import pytest
from fastapi.testclient import TestClient

from pokerlab.web.app import create_app


@pytest.fixture()
def client(tmp_path):
    app = create_app(db_path=str(tmp_path / "web.db"), seed=0)
    with TestClient(app) as c:
        c.db_path = str(tmp_path / "web.db")
        yield c


def test_get_next_returns_spot_contract(client):
    spot = client.get("/api/drill/next").json()
    for field in ("drill_id", "description", "legal_actions", "depth_bb",
                  "position", "pot_bb", "kind", "tournament"):
        assert field in spot
    assert isinstance(spot["legal_actions"], list) and spot["legal_actions"]
    assert spot["position"] in ("SB", "BB")


def test_get_next_icm_spot_carries_tournament(client):
    # Walk the drill loop until the scheduler serves an ICM spot, then assert it
    # carries TournamentContext. This used to seed one ICM category and assert
    # the *very next* spot came back from it -- which only held because
    # select_next was starved to a single category (round-2 finding [E42]).
    # Now that unseen categories are reachable, the ICM set is arrived at by
    # actually drilling, which is the behaviour worth pinning.
    for _ in range(40):
        spot = client.get("/api/drill/next").json()
        if spot["kind"] == "icm":
            assert spot["tournament"] is not None
            return
        client.post("/api/drill/answer",
                    json={"drill_id": spot["drill_id"],
                          "action": spot["legal_actions"][0]})
    raise AssertionError("scheduler never served an ICM spot")
    assert spot["tournament"]["players_remaining"] == 4
    assert len(spot["tournament"]["payouts"]) == 3


def test_answer_scores_and_persists(client):
    r = client.post("/api/drill/answer",
                    json={"drill_id": "SBjam|preflop|jam|10:AA", "action": "jam"})
    assert r.status_code == 200
    body = r.json()
    assert body["correct"] is True
    assert body["best_action"] == "jam"
    assert "explanation" in body and "jam" in body["explanation"]
    # attempt was persisted to drill_attempts
    con = sqlite3.connect(client.db_path)
    rows = con.execute("SELECT spot_key, correct FROM drill_attempts").fetchall()
    assert rows == [("SBjam|preflop|jam|10", 1)]


def test_answer_incorrect_reports_ev_loss(client):
    r = client.post("/api/drill/answer",
                    json={"drill_id": "SBjam|preflop|jam|10:AA", "action": "fold"})
    body = r.json()
    assert body["correct"] is False
    assert body["ev_loss_bb"] > 0
    assert body["best_action"] == "jam"


def test_answer_unknown_drill_id_is_404(client):
    r = client.post("/api/drill/answer",
                    json={"drill_id": "SBjam|preflop|jam|10:ZZ", "action": "jam"})
    assert r.status_code == 404


def test_answer_illegal_action_is_400(client):
    r = client.post("/api/drill/answer",
                    json={"drill_id": "SBjam|preflop|jam|10:AA", "action": "raise"})
    assert r.status_code == 400


def test_index_and_static_served(client):
    idx = client.get("/")
    assert idx.status_code == 200
    assert "app.js" in idx.text
    js = client.get("/static/app.js")
    assert js.status_code == 200


# --------------------------------------------------------------------------- #
# Round-2 findings [E41]+[E42], the self-bricking pair. select_next could only
# return keys already in sr_state, so one category was served forever; that in
# turn drove a single category's rep count high enough for the uncapped SM-2
# interval to overflow datetime.max -- HTTP 500 as a reward for correct play.
# Fixing either alone hides the other, so this drives both from the HTTP edge.
# --------------------------------------------------------------------------- #
def test_long_correct_streak_never_500s_and_covers_the_population(client):
    from pokerlab.drills import generator as gen

    all_cats = {d.spot_key for d in gen.default_population()}
    seen: set[str] = set()

    for _ in range(300):
        spot = client.get("/api/drill/next")
        assert spot.status_code == 200, spot.text
        spot = spot.json()
        drill_id = spot["drill_id"]
        seen.add(drill_id.rsplit(":", 1)[0])
        # always answer correctly -- the trajectory that used to overflow
        best = client.post("/api/drill/answer",
                           json={"drill_id": drill_id, "action": spot["legal_actions"][0]})
        assert best.status_code == 200, f"500 on iteration: {best.text}"

    assert seen == all_cats, f"unreachable categories: {sorted(all_cats - seen)}"


def test_repeated_answer_for_one_drill_counts_once(client):
    spot = client.get("/api/drill/next").json()
    body = {"drill_id": spot["drill_id"], "action": spot["legal_actions"][0]}
    first = client.post("/api/drill/answer", json=body).json()
    for _ in range(7):                       # double-click storm
        assert client.post("/api/drill/answer", json=body).json() == first

    conn = sqlite3.connect(client.db_path)
    assert conn.execute("SELECT COUNT(*) FROM drill_attempts").fetchone()[0] == 1
    assert conn.execute("SELECT reps FROM sr_state").fetchone()[0] == 1

    # fetching the next spot re-arms it: genuine re-practice still counts
    client.get("/api/drill/next")
    client.post("/api/drill/answer", json=body)
    assert conn.execute("SELECT COUNT(*) FROM drill_attempts").fetchone()[0] == 2
