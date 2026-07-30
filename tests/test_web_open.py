"""Open-game + resteal drills through the web layer.

The user-facing point of the whole slice: on an OPEN drill, "raise 2.2bb" is
a PRICED answer — graded with a real ev_loss — not an off-tree apology. On a
RESTEAL drill the hero faces a raise (not an all-in), and the scene draws
that difference from server facts: the raiser's seat shows the raise and the
stack still behind it.
"""

import pytest
from fastapi.testclient import TestClient

from pokerlab.charts.ring import RING_ORDER
from pokerlab.drills.generator import Drill
from pokerlab.types import Solution
from pokerlab.web.app import _action_line, _seats_json, _spot_json, create_app


def _resteal_drill() -> Drill:
    sol = Solution(actions={"jam": (1.0, 0.6), "fold": (-1.125, 0.4)},
                   range_ctx="chart:test", source="chart")
    return Drill(
        drill_id="BBresteal.vCO.r2.2|preflop|jam|10:A5s", kind="resteal",
        position="BB", depth_bb=10.0, hand_label="A5s", solution=sol,
        pot_bb=20.0, leak_key="BBresteal.vCO.r2.2|preflop|jam|10",
        legal_actions=("jam", "fold"),
        description="BB 10bb facing a CO raise to 2.2bb (9-max), A5s: "
                    "your action?",
        off_tree_actions=("call",), action_note="note",
        versus="CO", versus_action="raise 2.2bb", table=RING_ORDER)


def test_resteal_scene_shows_a_raise_not_an_allin():
    d = _resteal_drill()
    spot = _spot_json(d)
    assert spot["facing_allin"] is False           # a raise is not an all-in
    assert _action_line(d) == "CO raises to 2.2bb — action on you"
    co = next(s for s in spot["seats"] if s["pos"] == "CO")
    assert co["state"] == "raise"
    assert co["bet"] == "2.2bb"
    # the raiser's plate must not claim the full stack — 2.2bb of it is in
    # front; the server states what is BEHIND
    assert co["behind"] == pytest.approx(7.8)
    # seats between the raiser and the hero folded to reach this decision
    assert next(s for s in spot["seats"] if s["pos"] == "BTN")["state"] == "folded"


@pytest.fixture(scope="module")
def client():
    app = create_app(db_path=":memory:", seed=0)
    with TestClient(app) as c:
        yield c


def test_open_drill_prices_the_raise_for_real(client):
    r = client.post("/api/drill/answer", json={
        "drill_id": "COopen|preflop|open|20:A5s", "action": "raise 2.2bb"})
    assert r.status_code == 200
    body = r.json()
    assert body["ev_loss"] is not None      # PRICED — the whole point
    assert isinstance(body["ev_loss"], float)
    rta_actions = {a["action"] for a in body["rta"]["actions"]}
    assert {"jam", "raise 2.2bb", "raise 3bb", "fold"} == rta_actions


def test_open_drill_limp_stays_unpriced_with_the_why(client):
    # different hand from the test above: the replay guard absorbs an
    # immediately repeated drill_id by design
    r = client.post("/api/drill/answer", json={
        "drill_id": "COopen|preflop|open|20:A6s", "action": "limp"})
    assert r.status_code == 200
    assert r.json()["ev_loss"] is None


def test_jamfold_offtree_raise_points_at_the_open_mode(client):
    r = client.post("/api/drill/answer", json={
        "drill_id": "SBjam|preflop|jam|10:AA", "action": "raise 2.2bb"})
    body = r.json()
    assert body["ev_loss"] is None
    assert "first-in open" in body["explanation"]  # the bridge to real raises


def test_filter_serves_open_and_resteal_modes(client):
    spot = client.get("/api/drill/next?mode=open&pos=CO").json()
    assert spot["kind"] == "open" and spot["position"] == "CO"
    # 4 priced buttons above 5bb; at 5bb the raise menu is honestly empty
    # (sizes_for_depth), so jam/fold is the whole priced set there
    n = len(spot["legal_actions"])
    assert n == 4 or (n == 2 and spot["drill_id"].split("|")[3].startswith("5"))
    spot = client.get("/api/drill/next?mode=resteal&pos=BB").json()
    assert spot["kind"] == "resteal" and spot["position"] == "BB"


def test_resteal_drills_only_exist_where_the_raise_does(client):
    """UTG at 5bb never raises at equilibrium — its defend node is unreached
    and unsolved, so no resteal drill may exist there (grading against an
    unreached CFR infoset is grading against noise)."""
    from pokerlab.charts import hands
    from pokerlab.charts.openraise import open_solution
    from pokerlab.drills.generator import _MIN_RAISE_COMBOS, open_drills

    by_key = {}
    for d in open_drills():
        if d.kind == "resteal":
            by_key.setdefault(d.leak_key, d)
    for key, d in by_key.items():
        formation = key.split(".v")[1].split(".r")[0]
        rlab = d.versus_action
        sol = open_solution(formation, d.depth_bb,
                            _ante_of(d.leak_key))
        mass = sum(f * hands.combos(h) for f, h in
                   zip(sol.open_freq[rlab], hands.HAND_CLASSES))
        assert mass >= _MIN_RAISE_COMBOS, (key, mass)


def _ante_of(leak_key: str) -> float:
    token = leak_key.rsplit("|", 1)[1]
    return float(token.split("a")[1]) if "a" in token else 0.0
