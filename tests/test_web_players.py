"""The players filter (table-size axis, plan §5.1 ante-adjusted ranges row).

A category's drills share one formation, so each carries a players count —
a formation fact (len(table), or players_remaining for ICM). The filter
narrows the scheduler's vocabulary exactly like mode/pos; impossible
combinations 400 loudly (a filter that silently widens is lying)."""

import pytest
from fastapi.testclient import TestClient

from pokerlab.web.app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app(db_path=":memory:", seed=0)
    with TestClient(app) as c:
        yield c


def test_six_max_first_in_serves_a_six_handed_scene(client):
    spot = client.get("/api/drill/next?mode=ring-jam&players=6").json()
    assert spot["kind"] == "ring"
    assert len(spot["seats"]) == 6
    seats = [s["pos"] for s in spot["seats"]]
    assert seats == ["LJ", "HJ", "CO", "BTN", "SB", "BB"]


def test_players_2_is_the_heads_up_table(client):
    spot = client.get("/api/drill/next?players=2").json()
    assert len(spot["seats"]) == 2 or spot["kind"] == "river"


def test_impossible_combo_400s_loudly(client):
    r = client.get("/api/drill/next?mode=hu&players=9")
    assert r.status_code == 400
    assert "players" in r.json()["detail"]


def test_unknown_players_value_400s(client):
    assert client.get("/api/drill/next?players=11").status_code == 400
    assert client.get("/api/drill/next?players=x").status_code == 400


def test_short_ring_answers_grade_with_real_ev(client):
    spot = client.get("/api/drill/next?mode=ring-jam&players=4").json()
    r = client.post("/api/drill/answer", json={
        "drill_id": spot["drill_id"], "action": spot["legal_actions"][0]})
    assert r.status_code == 200
    assert isinstance(r.json()["ev_loss"], float)  # priced, not apologized


def test_app_js_sends_the_players_param():
    from pathlib import Path

    import pokerlab.web.app as webapp
    static = Path(webapp.__file__).parent / "static"
    assert "players=" in (static / "app.js").read_text()
    assert 'id="players"' in (static / "index.html").read_text()
