"""River drills through the web layer.

What tier 2 must LOOK like to the user: the page shows the real board and
the hero's concrete cards, labels the answer 'exact — solver-graded', and
carries the provenance sentence so the approximation (the fixture ranges,
not the solve) is disclosed on the page — server words, never client claims.
"""

import pytest
from fastapi.testclient import TestClient

from pokerlab.drills.river import river_drills
from pokerlab.hh.tiers import TIER_LABELS
from pokerlab.types import TIER_SOLVER
from pokerlab.web.app import _spot_json, create_app


@pytest.fixture(scope="module")
def client():
    app = create_app(db_path=":memory:", seed=0)
    with TestClient(app) as c:
        yield c


def test_river_spot_payload_shows_the_whole_scene():
    d = river_drills()[0]
    spot = _spot_json(d)
    assert spot["tier"] == TIER_SOLVER
    assert spot["tier_label"] == TIER_LABELS[TIER_SOLVER]
    assert len(spot["board"]) == 5                 # concrete cards, e.g. "As"
    assert spot["hero_cards"] and len(spot["hero_cards"]) == 2
    assert "flops25" in spot["provenance"]         # the disclosed inputs
    seats = {s["pos"] for s in spot["seats"]}
    assert seats == {"BB", "BTN"}


def test_preflop_spots_do_not_grow_a_board(client):
    spot = client.get("/api/drill/next?mode=hu").json()
    assert spot["board"] == []
    assert spot["hero_cards"] is None
    assert spot["provenance"] is None
    assert spot["tier"] == 1


def test_river_answer_grades_with_real_ev(client):
    d = river_drills()[0]
    best = max(d.solution.actions, key=lambda a: d.solution.actions[a][0])
    r = client.post("/api/drill/answer",
                    json={"drill_id": d.drill_id, "action": best})
    assert r.status_code == 200
    body = r.json()
    assert body["correct"] is True
    assert body["ev_loss"] == 0.0
    assert "Solver" in body["explanation"]


def test_filter_serves_river_mode(client):
    spot = client.get("/api/drill/next?mode=river").json()
    assert spot["kind"] == "river"
    assert len(spot["board"]) == 5
