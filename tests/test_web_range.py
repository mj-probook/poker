"""The range grid (plan §M2 note — deferred until drill answers were
range-shaped; the open game and table-size axis made them so).

The grid is served, not derived client-side, and it is built from the SAME
Solution objects that grade the drills (the population), so there is no
second source to drift from the answer keys. Every cell states the per-
action frequency of its 169-class; the action vocabulary and colors legend
come from the server payload.
"""

import pytest
from fastapi.testclient import TestClient

from pokerlab.web.app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app(db_path=":memory:", seed=0)
    with TestClient(app) as c:
        yield c


def test_range_grid_for_a_ring_category(client):
    r = client.get("/api/drill/range",
                   params={"drill_id": "COjam|preflop|jam|10:AA"})
    assert r.status_code == 200
    body = r.json()
    assert body["actions"] == ["jam", "fold"]
    grid = body["grid"]
    assert len(grid) == 13 and all(len(row) == 13 for row in grid)
    # canonical layout: diagonal pairs, upper suited, lower offsuit
    assert grid[0][0]["label"] == "AA"
    assert grid[0][1]["label"] == "AKs"
    assert grid[1][0]["label"] == "AKo"
    assert grid[12][12]["label"] == "22"
    # AA jams at 10bb first-in from the CO — the key's own frequency
    assert grid[0][0]["freq"]["jam"] > 0.95
    # the disclosure rides along: which game this range is FOR
    assert "single-caller" in body["range_ctx"]
    assert body["tier_label"].startswith("exact")


def test_range_grid_serves_the_open_game_action_set(client):
    r = client.get("/api/drill/range",
                   params={"drill_id": "COopen|preflop|open|20:A5s"})
    body = r.json()
    assert set(body["actions"]) == {"jam", "raise 2.2bb", "raise 3bb", "fold"}
    freqs = body["grid"][0][0]["freq"]           # AA at 20bb CO first-in
    assert abs(sum(freqs.values()) - 1.0) < 1e-4


def test_range_grid_for_river_marks_missing_classes_null(client):
    from pokerlab.drills.river import river_drills
    d = river_drills()[0]
    r = client.get("/api/drill/range", params={"drill_id": d.drill_id})
    body = r.json()
    cells = [c for row in body["grid"] for c in row]
    # the BB fixture range does not hold every class on a river — absent
    # classes are stated as null, never rendered as fold-100%
    assert any(c["freq"] is None for c in cells)
    assert any(c["freq"] is not None for c in cells)


def test_range_grid_unknown_drill_404s(client):
    assert client.get("/api/drill/range",
                      params={"drill_id": "nope:AA"}).status_code == 404


def test_app_js_renders_the_grid_from_server_words():
    from pathlib import Path

    import pokerlab.web.app as webapp
    js = (Path(webapp.__file__).parent / "static" / "app.js").read_text()
    assert "/api/drill/range" in js
    # revealed on demand AFTER answering — a visible grid before the answer
    # would leak the current hand's row
    assert "lastAnswer" in js.split("function showRange")[1].split("}")[0] \
        or "rangeBtn.disabled" in js
    html = (Path(webapp.__file__).parent / "static" / "index.html").read_text()
    assert 'id="range-panel"' in html
