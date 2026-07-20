"""Round-4 [1]-RENDER: the tier-2 approximation has to reach the leak report.

PLAN §5.3's rev-3.2 honesty note says the approximation "is recorded in each
Solution's provenance (`range_ctx`) AND in the leak report". The store half
landed it (756c50b); this is the surface half.

The distinction that makes the copy honest: a tier-2 ev_loss is EXACT for the
game it was handed — both ranges uniform, stacks symmetric — and not for the
hand as played. "Approximate" alone undersells it in the wrong direction: the
solve is exact, the inputs are the approximation.

Two failure modes this file exists to prevent, in opposite directions:

  * a blanket badge on the leak section — the f0c1bd8 defect class, where a
    label is asserted for every row on the strength of a property only some
    rows have. A leak_key MIXES tiers, so "3 of 11 decisions approximated" is
    the honest shape and a boolean is not available.
  * a row with nothing to disclose carrying a disclosure anyway. `provenance`
    is NULL for tier 1 and tier 3 on purpose — a chart grade's provenance IS
    the chart, and tier 3 has no reference solution. NULL means "nothing to
    disclose", never "unknown provenance".

The rendered disclosure is asserted against a SENTINEL provenance string
written into the DB, not against the shipped RANGE_PROVENANCE constant. With
one provenance string in the codebase, asserting the real one cannot tell a
data-driven render from a hardcoded literal — the [R4-10] lesson, applied
before it could bite rather than after.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import pokerlab.web.app as webapp
from pokerlab.store import db
from pokerlab.web.app import create_app

REPORT_JS = (Path(webapp.__file__).resolve().parent
             / "static" / "report.js").read_text()
AT = "2026-07-20T00:00:00"
# Deliberately not tier2.RANGE_PROVENANCE: the render must follow the DATA.
SENTINEL = "ranges=sentinel|stack=sentinel"
MIXED_LEAK = "BTNjam|preflop|jam|20"
EXACT_LEAK = "SBjam|preflop|jam|10"


@pytest.fixture
def served(tmp_path):
    """One leak with 1-of-2 decisions approximated, one fully exact."""
    dbp = str(tmp_path / "prov.db")
    conn = db.connect(dbp)
    hid = db.insert_imported_hand(conn, "PokerStars", "raw", "{}", AT, "h1")
    # A leak that MIXES an approximated tier-2 grade with an exact tier-1 one:
    # the case a boolean flag cannot describe honestly.
    db.insert_grading(conn, hid, 0, 2, "jam", "fold", 3.0, MIXED_LEAK, AT,
                      provenance=SENTINEL)
    db.insert_grading(conn, hid, 1, 1, "jam", "fold", 1.0, MIXED_LEAK, AT)
    # A leak with nothing to disclose at all.
    db.insert_grading(conn, hid, 2, 1, "jam", "fold", 9.0, EXACT_LEAK, AT)
    conn.close()
    with TestClient(create_app(db_path=dbp, seed=0)) as client:
        res = client.get("/api/report")
        assert res.status_code == 200
        return res.json()


def _row(body, leak_key):
    rows = body["leaks"]["rows"]
    return next(r for r in rows if r["leak_key"] == leak_key)


def test_an_approximated_leak_discloses_how_many_and_what_was_assumed(served):
    row = _row(served, MIXED_LEAK)
    assert row["decisions"] == 2
    # A COUNT, not a flag: one of the two came from an exact chart reference.
    assert row["approx_decisions"] == 1
    assert row["provenance"] == SENTINEL


def test_a_fully_exact_leak_discloses_nothing(served):
    """NULL provenance means nothing to disclose — it must not render as doubt."""
    row = _row(served, EXACT_LEAK)
    assert row["approx_decisions"] == 0
    assert row["provenance"] is None


def test_the_caveat_is_the_servers_words_and_only_the_servers(served):
    """The page renders the explanation; it never authors or restates it.

    Same rule as tier3.label and IN_FLIGHT_LABEL. A page that writes its own
    version of "what approximated means" is a second home for a claim the
    grader owns, and it drifts silently the day the assumption changes.
    """
    assert served["leaks"]["approx_caveat"] == webapp.APPROX_CAVEAT
    assert webapp.APPROX_CAVEAT not in REPORT_JS


def test_the_caveat_states_the_solve_is_exact_for_the_inputs_it_got(served):
    """Not merely "approximate" — that undersells it in the wrong direction.

    The accuracy bar applies to the solve GIVEN those inputs (PLAN §5.3 rev
    3.2). Copy that says only "approximate" invites the reader to discount the
    number as sloppy, when what is actually approximate is the game it was
    handed. Pinned as a property of the served sentence, since this is the one
    claim the whole item exists to make.
    """
    caveat = served["leaks"]["approx_caveat"].lower()
    assert "exact" in caveat, "the solve's exactness must survive the caveat"
    assert "uniform" in caveat or "range" in caveat, "name what was assumed"
