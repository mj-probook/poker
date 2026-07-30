"""Practice filter: choose what to drill instead of waiting for the queue.

The scheduler is SM-2-first (due leaks outrank everything), which is right
for training but wrong for "show me a CO spot NOW" — the 2026-07-22 report:
the user merged 9-max ring drills and could not reach one on demand. The
filter narrows the CATEGORY vocabulary the scheduler sees; within the
filtered pool SM-2 still rules (due first, then unseen, interleaved).

Impossible combinations fail loudly (400 naming the problem), never fall
back silently to unfiltered — a filter that quietly widens itself is lying.
"""

import pytest
from fastapi.testclient import TestClient

from pokerlab.web.app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app(db_path=":memory:", seed=0)
    with TestClient(app) as c:
        yield c


def test_ring_jam_mode_serves_only_first_in_ring_spots(client):
    for _ in range(6):
        spot = client.get("/api/drill/next?mode=ring-jam").json()
        assert spot["kind"] == "ring"
        assert spot["facing_allin"] is False
        assert spot["legal_actions"] == ["jam", "fold"]


def test_position_filter_pins_the_hero_seat(client):
    for _ in range(4):
        spot = client.get("/api/drill/next?mode=ring-jam&pos=CO").json()
        assert spot["position"] == "CO"


def test_defend_mode_with_position(client):
    spot = client.get("/api/drill/next?mode=ring-defend&pos=BB").json()
    assert spot["kind"] == "ring"
    assert spot["position"] == "BB"
    assert spot["facing_allin"] is True


def test_hu_and_icm_modes(client):
    assert client.get("/api/drill/next?mode=hu").json()["kind"] == "jamfold"
    assert client.get("/api/drill/next?mode=icm").json()["kind"] == "icm"


def test_unknown_mode_is_400(client):
    assert client.get("/api/drill/next?mode=turbo").status_code == 400


def test_impossible_combo_is_400_not_silent_widening(client):
    # UTG is first to act — it never defends a jam, so this pool is empty
    r = client.get("/api/drill/next?mode=ring-defend&pos=UTG")
    assert r.status_code == 400
    assert "no drills" in r.json()["detail"].lower()


def test_default_stays_unfiltered(client):
    assert client.get("/api/drill/next").status_code == 200
