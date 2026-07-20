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
    #
    # The walk budget is DERIVED from the vocabulary, not a constant: the ICM
    # set is 2 categories out of N, so the number of drills needed to reach it
    # scales with N. A hardcoded 40 silently went stale the moment the ante
    # buckets tripled the vocabulary from 12 to 32 (round-3 finding [E49]) --
    # ICM then first appeared at iteration 50 and this read as a scheduler
    # regression. Deriving it means the next vocabulary change cannot repeat
    # that.
    from pokerlab.drills import generator as gen
    budget = 4 * len({d.leak_key for d in gen.default_population()})
    for _ in range(budget):
        spot = client.get("/api/drill/next").json()
        if spot["kind"] == "icm":
            assert spot["tournament"] is not None
            return
        client.post("/api/drill/answer",
                    json={"drill_id": spot["drill_id"],
                          "action": spot["legal_actions"][0]})
    raise AssertionError("scheduler never served an ICM spot")


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
    rows = con.execute("SELECT leak_key, correct FROM drill_attempts").fetchall()
    assert rows == [("SBjam|preflop|jam|10", 1)]


def test_answer_incorrect_reports_ev_loss(client):
    r = client.post("/api/drill/answer",
                    json={"drill_id": "SBjam|preflop|jam|10:AA", "action": "fold"})
    body = r.json()
    assert body["correct"] is False
    # `ev_loss` + `ev_unit`, not `ev_loss_bb`: the old key asserted the unit in
    # its NAME, which was false for ICM drills ([P7']).
    assert body["ev_loss"] > 0
    assert body["ev_unit"] == "bb"          # this one IS a chip-EV drill
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

    all_cats = {d.leak_key for d in gen.default_population()}
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


def test_one_category_driven_past_the_old_overflow_point(client):
    """[E41] the docs-reviewer path: hammer a SINGLE category to rep 14+.

    The 300-drill test above rotates across categories, so no single sr_state
    row climbs far -- it proves coverage, not the cap. This drives one category
    only, which is the trajectory that actually overflowed (rep 14 raised
    OverflowError and 500'd). Different hand classes within the one category
    sidestep the repeat-answer guard, so every POST is a real review.
    """
    import sqlite3

    from pokerlab.drills import generator as gen

    cat = "SBjam|preflop|jam|5"
    drills = [d for d in gen.default_population() if d.leak_key == cat]
    assert len(drills) >= 30, "need distinct drills in one category"

    for drill in drills[:30]:                       # 30 > the old rep-14 cliff
        best = max(drill.solution.actions,
                   key=lambda a: drill.solution.actions[a][0])
        r = client.post("/api/drill/answer",
                        json={"drill_id": drill.drill_id, "action": best})
        assert r.status_code == 200, f"{drill.drill_id}: {r.status_code} {r.text}"
        client.get("/api/drill/next")               # re-arm the repeat guard

    conn = sqlite3.connect(client.db_path)
    reps, interval, ef = conn.execute(
        "SELECT reps, interval_days, easiness FROM sr_state WHERE leak_key=?",
        (cat,)).fetchone()
    assert reps == 30                                # all 30 counted
    assert interval <= 365.0 and ef <= 3.0           # both ceilings held


# --------------------------------------------------------------------------- #
# Round-3 finding [P7']: the feedback line reported every drill's EV loss with
# a "bb" suffix, including ICM drills whose ev_loss is an ICM-$ delta ~25x
# larger per unit. An ICM loss of 100 read as a catastrophic 100bb error rather
# than the $100 of a 1000$ pool that it is.
# --------------------------------------------------------------------------- #
def test_icm_feedback_is_labelled_in_icm_dollars_not_bb(client):
    from pokerlab.drills import generator as gen

    icm = gen.icm_drills()[0]
    r = client.post("/api/drill/answer",
                    json={"drill_id": icm.drill_id, "action": "fold"})
    body = r.json()
    assert body["ev_unit"] == "ICM-$"
    assert "ICM-$" in body["explanation"]
    assert "bb)" not in body["explanation"], "ICM loss still labelled as bb"


def test_chip_ev_feedback_is_still_labelled_in_bb(client):
    """Guards the test above from 'fixing' the label by relabelling everything."""
    r = client.post("/api/drill/answer",
                    json={"drill_id": "SBjam|preflop|jam|10:AA", "action": "fold"})
    body = r.json()
    assert body["ev_unit"] == "bb"
    assert "bb)" in body["explanation"] and "ICM-$" not in body["explanation"]


def test_every_drill_kind_has_a_declared_ev_unit():
    """A new drill kind must surface loudly, not silently format as bb.

    Mirrors the builder's TERMINAL_ACTIONS/BATCH_STATUSES pin: the generator
    owns the kind vocabulary, app.py owns only the operator-facing copy, and
    this asserts the two agree.
    """
    from pokerlab.drills import generator as gen
    from pokerlab.web.app import EV_UNITS

    kinds = {d.kind for d in gen.default_population()}
    assert kinds <= set(EV_UNITS), f"drill kinds with no declared unit: {kinds - set(EV_UNITS)}"


def test_units_and_tier_are_independent_axes(client):
    """ICM drills are tier-1 chart-graded AND denominated in $; both, not either.

    Folding units into the tier vocabulary would let a future drill kind
    inherit one claim by asserting the other.
    """
    from pokerlab.drills import generator as gen

    icm = gen.icm_drills()[0]
    for _ in range(200):
        spot = client.get("/api/drill/next").json()
        if spot["kind"] == "icm":
            assert spot["tier"] == 1                    # unchanged by [P7']
            assert spot["ev_unit"] == "ICM-$"           # but the unit differs
            return
        client.post("/api/drill/answer",
                    json={"drill_id": spot["drill_id"],
                          "action": spot["legal_actions"][0]})
    raise AssertionError("scheduler never served an ICM spot")


def test_the_route_grades_icm_in_icm_dollars_not_just_labels_it(client):
    """The ROUTE must pass the conversion, not merely name the unit.

    Labelling and grading are separate wires and only one of them was visible
    in the feedback string: with `bb_value` dropped at the call site the
    explanation still reads "ICM-$" while the verdict silently reverts to
    exact-argmax. Every other test here passed with that wire cut, so this
    pins the wire itself -- a spot 0.37 ICM-$ off the best action, which is
    inside the converted eps of 2.50 and far outside the unconverted 0.10.
    """
    r = client.post("/api/drill/answer",
                    json={"drill_id": "BBcall.icm|preflop|call|10:A7s",
                          "action": "fold"})
    body = r.json()
    assert body["ev_loss"] == pytest.approx(0.37, abs=0.01)
    assert body["correct"] is True, (
        "a 0.37 ICM-$ deviation was graded wrong — the route is comparing a "
        "$-delta against the unconverted bb epsilon")
