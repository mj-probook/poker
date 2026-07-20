"""Round-3 finding [P5'] (P1): the training loop had no report surface.

Everything the loop produced — the session's failed/pending counts, the ranked
leaks, the tier-3 deviations, the skill gates — was reachable only from Python.
Plan §9 makes the tier labels load-bearing UI ("not fine print"), so the
separation between exact-tier EV-loss and approximate tier-3 frequency flags is
asserted here as a property of the SERVED payload, not of a docstring.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pokerlab.hh.ggpoker import parse_ggpoker
from pokerlab.hh.persist import persist_session
from pokerlab.hh.pokerstars import parse_pokerstars
from pokerlab.hh.population import load_population
from pokerlab.hh.report import grade_session
from pokerlab.store import db
import pokerlab.web.app as webapp
from pokerlab.web.app import create_app

FIXTURES = Path(__file__).parent / "fixtures" / "hh"
STATIC = Path(webapp.__file__).resolve().parent / "static"
AT = "2026-07-19T00:00:00"


@pytest.fixture
def imported(tmp_path):
    """A real imported session behind a live app."""
    dbp = str(tmp_path / "report.db")
    files = ["ps_multiway_flop.txt", "ps_ante_hu.txt", "gg_sb_fold_leak.txt"]
    raws = [(FIXTURES / f).read_text() for f in files]
    parsed = [parse_ggpoker(r) if f.startswith("gg") else parse_pokerstars(r)
              for f, r in zip(files, raws)]
    conn = db.connect(dbp)
    report = grade_session(parsed, population=load_population())
    persist_session(conn, parsed, report, graded_at=AT, raw_texts=raws)
    conn.close()
    return dbp


def _report(dbp):
    with TestClient(create_app(db_path=dbp, seed=0)) as client:
        res = client.get("/api/report")
        assert res.status_code == 200
        return res.json()


def test_report_summarises_the_session_including_failures_and_backlog(imported):
    s = _report(imported)["session"]
    assert s["hands"] == 3
    assert s["graded"] > 0
    # plan §5.3: a session is PARTIAL until its tier-2 backlog is solved, and
    # the operator must be told so rather than reading a complete-looking report
    assert s["queued"] > 0 and s["partial"] is True
    assert "failed" in s


def test_report_ranks_the_exact_tier_leaks_by_ev_loss(imported):
    leaks = _report(imported)["leaks"]
    assert 0 < len(leaks) <= 5
    assert [r["ev_loss_per_100"] for r in leaks] == sorted(
        (r["ev_loss_per_100"] for r in leaks), reverse=True)
    # every ranked leak came from a tier that HAS an EV oracle
    assert all(r["ev_loss_per_100"] is not None for r in leaks)


def test_tier3_is_listed_separately_and_labelled_approximate(imported):
    body = _report(imported)
    t3 = body["tier3"]

    # the label is part of the PAYLOAD — the honesty claim travels with the data
    assert "approximate" in t3["label"].lower()
    assert "no ev loss" in t3["label"].lower()

    assert t3["rows"], "fixture session has a multiway decision"
    for row in t3["rows"]:
        assert "ev_loss" not in row and "ev_loss_per_100" not in row
    # and it is never mixed into the ranking
    ranked = {r["leak_key"] for r in body["leaks"]}
    assert not (ranked & {r["leak_key"] for r in t3["rows"]})


def test_report_carries_both_skill_gates(imported):
    gates = _report(imported)["gates"]
    assert "ev_loss_trend" in gates and "accuracy_by_kind" in gates
    assert all({"bucket", "leak_key", "ev_loss_per_100"} <= set(r)
               for r in gates["ev_loss_trend"])


def test_report_page_is_served_and_renders_both_tiers(imported):
    with TestClient(create_app(db_path=imported, seed=0)) as client:
        html = client.get("/report").text
    assert "/static/report.js" in html
    for slot in ("summary", "leaks", "tier3", "gates"):
        assert f'id="{slot}"' in html


def test_report_js_renders_the_servers_tier_label_rather_than_its_own():
    """The label must reach the page — but the page must not author it.

    Asserting the word "approximate" appears in report.js would demand the JS
    hard-code the honesty claim, which is the drift this design prevents: the
    label lives once, in hh.tiers.TIER_LABELS, travels in the payload, and the
    page renders whatever the grader said. So the invariant is structural.
    """
    js = (STATIC / "report.js").read_text()
    assert "/api/report" in js
    assert "tier3.label" in js, "the served label is what gets rendered"
    assert "population_frequency" in js
    # a missing baseline is "unknown", never a rendered 0%
    assert "no baseline" in js


# --------------------------------------------------------------------------- #
# [P5'] second half: a drill has to say which tier grades it, for the same
# reason the report does — plan §9, tier labels are load-bearing UI.
# --------------------------------------------------------------------------- #
def test_every_drill_spot_carries_its_tier_label(imported):
    with TestClient(create_app(db_path=imported, seed=0)) as client:
        spot = client.get("/api/drill/next").json()
    assert spot["tier"] == 1                       # chart-graded, exact
    assert "exact" in spot["tier_label"].lower()
    assert "chart" in spot["tier_label"].lower()
