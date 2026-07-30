"""Execute the shipped renderer against real payloads (DOM stub, no browser).

The text-contract tests (test_web_static) prove the JS mentions the right
field names; this suite proves render() actually RUNS on the three new spot
shapes — open (priced raise buttons), resteal (raise badge + behind stack),
river (board + concrete hero cards + provenance) — and that every element id
the renderer touches exists in the shipped index.html. A renderer reaching
for a slot the page doesn't have is invisible to text checks and an instant
blank screen in the browser.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import pokerlab.web.app as webapp
from pokerlab.web.app import _spot_json

STATIC = Path(webapp.__file__).resolve().parent / "static"
STUB = Path(__file__).parent / "fixtures" / "dom_stub_render.js"


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available to execute the shipped JS")
    from pokerlab.drills.generator import open_drills
    from pokerlab.drills.river import river_drills

    opens = open_drills()
    spots = {
        "open": _spot_json(next(d for d in opens if d.kind == "open"
                                and d.depth_bb == 20.0)),
        "resteal": _spot_json(next(d for d in opens if d.kind == "resteal")),
        "river": _spot_json(river_drills()[0]),
    }
    html = (STATIC / "index.html").read_text()
    import re
    ids = re.findall(r'id="([^"]+)"', html)

    tmp = tmp_path_factory.mktemp("dom")
    (tmp / "spots.json").write_text(json.dumps(spots))
    (tmp / "ids.json").write_text(json.dumps(ids))
    out = subprocess.run(
        [node, str(STUB), str(STATIC / "app.js"),
         str(tmp / "spots.json"), str(tmp / "ids.json")],
        capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1]), spots


def test_open_spot_renders_priced_raise_buttons(rendered):
    out, spots = rendered
    r = out["open"]
    assert set(spots["open"]["legal_actions"]) <= set(r["buttons"])
    assert "limp" in r["buttons"]            # the distractor, indistinguishable
    assert r["board_cards"] == 0             # preflop: no board grows
    assert r["provenance"] == ""             # tier 1: no provenance sentence


def test_resteal_spot_renders_the_raise_not_an_allin(rendered):
    out, spots = rendered
    r = out["resteal"]
    spot = spots["resteal"]
    assert "raises to" in r["action_line"]
    # the raiser's badge (bet amount) and behind-stack reach the seat markup
    bet = next(s["bet"] for s in spot["seats"] if s.get("state") == "raise")
    assert bet in r["seats_dump"]
    assert "rbet" in r["seats_dump"]
    assert "ALL-IN" not in r["seats_dump"]


def test_river_spot_renders_board_concrete_cards_and_provenance(rendered):
    out, spots = rendered
    r = out["river"]
    assert r["board_cards"] == 5
    assert r["hole_cards"] == 2
    # concrete suits from the payload, not synthetic display suits: both hero
    # card strings map to their real glyphs in the rendered hole cards
    glyph = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}
    for cs in spots["river"]["hero_cards"]:
        assert glyph[cs[1]] in r["hole_dump"]
    assert "flops25" in r["provenance"]
    assert "solver-graded" in r["meta_dump"]
    assert {"check", "jam"} <= set(r["buttons"])
