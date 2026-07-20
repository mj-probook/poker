"""Round-4 [9]: `partial` was doing double duty, and the two duties disagree.

PLAN §5.3 marks a session PARTIAL until *its* batch is solved. The CLI has
always been faithful to that — it prints its banner from `counts["queued"]`,
the rows THIS import enqueued. The API computed the same flag from a DB-wide
in-flight count, so any leftover backlog from an earlier session made every
later session read partial, including sessions that queued nothing at all.

Two surfaces, one word, opposite answers about the same session — and nothing
compared them, because no test in either suite exercised the partial path.
This file is that comparison, asserted as an equivalence rather than as two
independent expectations: pinning each surface separately is what let them
drift, since both were individually self-consistent.

The DB-wide number is not deleted, only renamed. It answers a real question
("is the solver behind on anything at all?") and losing it would trade one
wrong answer for a missing one. It just may not be called `partial`, because
the defect was the WORD carrying two scopes, not the count existing.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import pokerlab.web.app as webapp
from pokerlab.cli import main_import
from pokerlab.web.app import IN_FLIGHT_LABEL, create_app

FIXTURES = Path(__file__).parent / "fixtures" / "hh"
REPORT_JS = (Path(webapp.__file__).resolve().parent
             / "static" / "report.js").read_text()
# Routes 4 decisions to tier 2, so importing it enqueues a real backlog.
BACKLOGGED = str(FIXTURES / "ps_ante_hu.txt")
# Routes only tier 1: graded on arrival, nothing queued, nothing to drain.
CLEAN = str(FIXTURES / "gg_sb_fold_leak.txt")

CLI_BANNER = "session is PARTIAL"


def _api_session(dbp):
    with TestClient(create_app(db_path=dbp, seed=0)) as client:
        res = client.get("/api/report")
        assert res.status_code == 200
        return res.json()["session"]


def _import(path, dbp, capsys):
    """Run one import; return (cli_said_partial, api_session) for THAT session."""
    assert main_import([path, "--db", dbp]) == 0
    said = CLI_BANNER in capsys.readouterr().out
    return said, _api_session(dbp)


def test_cli_banner_and_api_partial_agree_on_a_backlogged_session(
        tmp_path, capsys) -> None:
    dbp = str(tmp_path / "backlog.db")
    said, session = _import(BACKLOGGED, dbp, capsys)

    assert said, "fixture must actually queue tier-2 work for this pin to mean anything"
    assert session["partial"] == said


def test_cli_banner_and_api_partial_agree_on_a_clean_session(
        tmp_path, capsys) -> None:
    """The divergent case: clean session, stale backlog from an earlier one.

    This is the whole defect. Session 1 leaves four pending rows; session 2
    queues nothing and the CLI correctly prints no banner. The API used to
    answer `partial` from the DB-wide count and call session 2 partial anyway
    — telling the operator to run a drain that would not change one number in
    session 2's report.
    """
    dbp = str(tmp_path / "mixed.db")
    _import(BACKLOGGED, dbp, capsys)          # leaves in-flight rows behind
    said, session = _import(CLEAN, dbp, capsys)

    assert not said, "the clean fixture must queue nothing, or this proves nothing"
    assert session["partial"] == said, (
        "a session that queued nothing is not partial, however far behind the "
        "solver is on OTHER sessions")


def test_the_db_wide_backlog_survives_under_its_own_name(
        tmp_path, capsys) -> None:
    """Renamed, not deleted — and not readable as a claim about this session.

    Kept separate from the two equivalence pins because the wrong fix for them
    is to drop the DB-wide count entirely, which would go green on both while
    losing the operator's only view of total solver backlog.
    """
    dbp = str(tmp_path / "wide.db")
    _import(BACKLOGGED, dbp, capsys)
    _, session = _import(CLEAN, dbp, capsys)

    assert session["in_flight_total"] == 4, "the earlier session's rows still exist"
    assert not session["partial"], "but they are not THIS session's problem"
    # The scope travels WITH the number, as the server's own words — the same
    # rule as tier3.label. Asserted as identity with the server constant rather
    # than by banning a word: the clearest label available says "not only this
    # session", so a keyword ban would have forbidden the best wording and
    # passed a vague one. The page must not be able to author this string.
    assert session["in_flight_label"] == IN_FLIGHT_LABEL
    assert IN_FLIGHT_LABEL not in REPORT_JS, (
        "report.js must render the served label, never restate the scope itself")
