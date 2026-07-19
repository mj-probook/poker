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
    # Answering an ICM drill seeds its category into the scheduler; the next
    # spot then comes from that ICM category and must carry TournamentContext.
    client.post("/api/drill/answer",
                json={"drill_id": "SBjam.icm|preflop|jam|10:AA", "action": "jam"})
    spot = client.get("/api/drill/next").json()
    assert spot["kind"] == "icm"
    assert spot["tournament"] is not None
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
