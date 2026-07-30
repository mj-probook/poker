"""Slice E — static drill-loop contract (plan §3 UI; no browser automation).

Rather than drive a browser, assert the shipped JS talks to exactly the routes
and JSON fields the API serves — a cheap, deterministic contract check that the
front end and back end agree.
"""

from pathlib import Path

import pytest

import pokerlab.web.app as webapp

STATIC = Path(webapp.__file__).resolve().parent / "static"


def test_app_js_fetches_the_exact_api_routes():
    js = (STATIC / "app.js").read_text()
    assert "/api/drill/next" in js
    assert "/api/drill/answer" in js
    assert '"POST"' in js or "'POST'" in js


def test_app_js_sends_and_reads_the_contract_fields():
    js = (STATIC / "app.js").read_text()
    # request payload fields
    assert "drill_id" in js and "action" in js
    # response/spot fields it renders
    for field in ("description", "legal_actions", "tournament", "correct",
                  "explanation"):
        assert field in js, field


def test_index_loads_the_app_script_and_action_slots():
    html = (STATIC / "index.html").read_text()
    assert "/static/app.js" in html
    assert 'id="actions"' in html
    assert 'id="feedback"' in html


def test_app_js_checks_response_status_before_rendering():
    """[15] A failed fetch rendered `undefined` — the error detail was dropped.

    Both fetches must check res.ok and surface the server's error instead of
    letting an error body flow into the happy path.
    """
    js = (STATIC / "app.js").read_text()
    assert js.count("res.ok") >= 2, "both fetches must check response status"
    # the server's error payload is what gets shown, not `undefined`
    assert "detail" in js


def test_app_js_renders_the_payout_ladder():
    """[P8'] An ICM spot is unanswerable without the prize ladder.

    The payload has carried `payouts` all along and the page dropped it,
    rendering only "N left" and the stacks. Identical stacks play completely
    differently on a flat ladder versus a top-heavy one — that shape is what
    ICM prices, and the answer key uses it — so withholding it asked the user
    to solve for information the drill was holding back.
    """
    js = (STATIC / "app.js").read_text()
    assert "payouts" in js, "app.js never reads tc.payouts"
    assert "ordinal" in js, "ladder places are not rendered as 1st/2nd/3rd"


def test_ordinal_helper_is_correct_including_the_11_to_13_irregulars():
    """Executes the shipped helper rather than grepping for digits.

    A substring check ("11" in js) would pass on any file containing those
    characters anywhere — it asserts nothing about behaviour. Running the real
    function is the only way to catch `11st`, the bug every naive
    `n % 10` ordinal has. Skipped, loudly, where node is unavailable: a check
    that cannot run should say so rather than silently report green (the
    round-2 [E52] lesson).
    """
    import json
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available to execute the shipped JS")

    # Extract just the helper: app.js touches `document` at module scope, so
    # the whole file cannot be evaluated headlessly, and restructuring shipped
    # code to suit a test is the wrong trade.
    js = (STATIC / "app.js").read_text()
    start = js.index("function ordinal(")
    depth, end = 0, None
    for i in range(js.index("{", start), len(js)):
        depth += (js[i] == "{") - (js[i] == "}")
        if depth == 0:
            end = i + 1
            break
    assert end is not None, "could not delimit the ordinal() helper"
    helper = js[start:end]

    cases = [1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 101, 111, 112]
    expected = ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st",
                "22nd", "23rd", "101st", "111th", "112th"]
    script = f"{helper}\nconsole.log(JSON.stringify({cases}.map(ordinal)));"
    out = subprocess.run([node, "-e", script], capture_output=True, text=True,
                         timeout=30)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1]) == expected


def test_app_js_renders_the_postflop_scene_from_server_facts():
    """River drills ship a real board and concrete hero cards; the page must
    draw exactly those server facts — board slot, concrete-suit mapping,
    provenance sentence — and offer the check/bet buttons in poker order."""
    js = (STATIC / "app.js").read_text()
    for field in ("spot.board", "spot.hero_cards", "spot.provenance"):
        assert field in js, field
    # concrete suit chars map to glyphs (synthetic display suits would
    # misstate a postflop spot — the suits are the strategy on a board)
    for glyph in ("♣", "♦", "♥", "♠"):
        assert glyph in js
    # postflop actions have an order slot like every other button
    for action in ('"check"', '"bet 33%"', '"bet 75%"'):
        assert action in js, action
    # the river practice mode exists, hero-side only (BB is the root actor)
    assert '"river": ["BB"]' in js


def test_index_has_the_postflop_slots():
    html = (STATIC / "index.html").read_text()
    assert 'id="board"' in html
    assert 'id="provenance"' in html
    assert 'value="river"' in html
