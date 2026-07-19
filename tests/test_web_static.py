"""Slice E — static drill-loop contract (plan §3 UI; no browser automation).

Rather than drive a browser, assert the shipped JS talks to exactly the routes
and JSON fields the API serves — a cheap, deterministic contract check that the
front end and back end agree.
"""

from pathlib import Path

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
