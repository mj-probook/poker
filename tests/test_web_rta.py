"""Off-tree action buttons + the RTA (post-answer) panel.

The drill's solved game is jam/fold — the chart has no EV for "raise 2.2bb".
So extra buttons are DISTRACTORS: submittable, graded incorrect as a framework
deviation with a server-authored explanation, and persisted with ev_loss NULL
(the tier-3 pattern: no priced number exists, and fabricating 0.0 would claim
a wrong action cost nothing). BB spots facing an all-in get NO distractors:
poker itself allows only call/fold there, and drawing impossible actions would
render a false fact.

The RTA panel is built from real solver output only — per-action EV and
frequency from the Solution, the decision-ε actually used, bb_value — so the
page can explain the answer in server words without authoring claims.
"""

import sqlite3

import pytest
from fastapi.testclient import TestClient

from pokerlab.web.app import create_app


@pytest.fixture()
def client(tmp_path):
    app = create_app(db_path=str(tmp_path / "rta.db"), seed=0)
    with TestClient(app) as c:
        c.db_path = str(tmp_path / "rta.db")
        yield c


SB_DRILL = "SBjam|preflop|jam|10:AA"      # SB open spot: distractors apply
BB_DRILL = "BBcall|preflop|call|10:AA"    # facing an all-in: call/fold only


def _spot(client, drill_id):
    # deterministic fetch: answer endpoint knows every drill by id, and the
    # spot fields come from the same Drill via _spot_json — exercise both.
    r = client.post("/api/drill/answer",
                    json={"drill_id": drill_id, "action": "fold"})
    assert r.status_code == 200
    return r.json()


def test_sb_spot_offers_off_tree_actions(client):
    # find an SB spot via the generator-backed payload contract
    from pokerlab.drills import generator as gen
    from pokerlab.web.app import _spot_json
    sb = next(d for d in gen.default_population() if d.position == "SB")
    spot = _spot_json(sb)
    assert len(spot["off_tree_actions"]) >= 3
    assert set(spot["off_tree_actions"]).isdisjoint(set(spot["legal_actions"]))


def test_bb_spot_has_no_off_tree_actions_and_says_why(client):
    from pokerlab.drills import generator as gen
    from pokerlab.web.app import _spot_json
    bb = next(d for d in gen.default_population() if d.position == "BB")
    spot = _spot_json(bb)
    assert spot["off_tree_actions"] == []
    # the reason is a server-authored sentence, present and non-empty
    assert spot["action_note"]


def test_off_tree_answer_is_graded_not_400(client):
    r = client.post("/api/drill/answer",
                    json={"drill_id": SB_DRILL, "action": "raise 2.2bb"})
    assert r.status_code == 200
    body = r.json()
    assert body["correct"] is False
    assert body["ev_loss"] is None          # no priced EV — never fabricated
    # the explanation must state the priced answer so the user learns the
    # chart's action, not only that theirs was off-tree
    assert body["best_action"] in body["explanation"]


def test_bare_unknown_action_is_still_400(client):
    r = client.post("/api/drill/answer",
                    json={"drill_id": SB_DRILL, "action": "raise"})
    assert r.status_code == 400


def test_off_tree_on_bb_spot_is_400(client):
    # distractors are per-spot: a BB spot never declared them, so they stay
    # unknown actions there
    r = client.post("/api/drill/answer",
                    json={"drill_id": BB_DRILL, "action": "raise 2.2bb"})
    assert r.status_code == 400


def test_answer_carries_rta_panel_from_real_solver_output(client):
    body = _spot(client, SB_DRILL)
    rta = body["rta"]
    acts = {row["action"]: row for row in rta["actions"]}
    assert set(acts) == {"jam", "fold"}     # exactly the solved game's actions
    for row in acts.values():
        assert isinstance(row["ev"], float)
        assert 0.0 <= row["frequency"] <= 1.0
    assert rta["best_action"] == max(acts, key=lambda a: acts[a]["ev"])
    assert rta["epsilon"] > 0
    assert rta["reasoning"]                 # server-authored, non-empty


def test_off_tree_answer_also_carries_rta_panel(client):
    r = client.post("/api/drill/answer",
                    json={"drill_id": SB_DRILL, "action": "limp"})
    body = r.json()
    assert set(row["action"] for row in body["rta"]["actions"]) == {"jam", "fold"}


def test_off_tree_attempt_persists_with_null_ev_loss(client):
    client.post("/api/drill/answer",
                json={"drill_id": SB_DRILL, "action": "limp"})
    conn = sqlite3.connect(client.db_path)
    row = conn.execute(
        "SELECT chosen, correct, ev_loss FROM drill_attempts"
        " ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    assert row == ("limp", 0, None)


def test_on_tree_attempt_still_persists_its_ev_loss(client):
    client.post("/api/drill/answer",
                json={"drill_id": SB_DRILL, "action": "fold"})
    conn = sqlite3.connect(client.db_path)
    row = conn.execute(
        "SELECT chosen, ev_loss FROM drill_attempts"
        " ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    assert row[0] == "fold" and row[1] is not None
