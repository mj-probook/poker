"""Round-4 [R2' follow-on]: a failed hand that CAN be named must be named.

PLAN §8's M4 exit promises failed hands are "surfaced in the session summary —
never silently dropped". Surfacing a hand the operator cannot identify only
half-keeps that promise: "1 hand could not be graded" is a fact you can read
and cannot act on. The remedy for a bad hand is to go find it in the session
file, and the hand number is how you do that.

The two parse-stage failures are NOT the same case, and this file pins the
difference in both directions:

  * the HEADER fails to parse -> there is no hand number to report, and
    `UNIDENTIFIED_HAND` is the honest rendering. Inventing one would be worse
    than saying nothing.
  * the header parses and the BODY fails -> the hand number was read
    successfully and is sitting in the stored raw text. Reporting *that* hand
    as unidentified is a self-inflicted loss of information.

Asserted in both directions on purpose: an implementation that simply never
renders "unidentified" would satisfy the second claim while breaking the first,
and the first is the one that keeps us from inventing hand numbers.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pokerlab.cli import main_import
from pokerlab.store import db
from pokerlab.web.app import UNIDENTIFIED_HAND, create_app

FIXTURES = Path(__file__).parent / "fixtures" / "hh"
# Two hands: the first grades, the second has a well-formed header
# (#240000000099) and no table line, so it dies in the BODY.
BODY_FAILS = str(FIXTURES / "ps_body_fails.txt")
# The middle chunk's header is "PokerStars Hand #BROKEN" — nothing to name.
HEADER_FAILS = str(FIXTURES / "ps_session_mixed.txt")
BODY_FAILS_UID = "240000000099"


def _failed_hands(dbp):
    with TestClient(create_app(db_path=dbp, seed=0)) as client:
        res = client.get("/api/report")
        assert res.status_code == 200
        return res.json()["session"]["failed_hands"]


def test_body_failure_is_reported_by_its_real_hand_number(tmp_path) -> None:
    """The header parsed, so the report must say WHICH hand was lost."""
    dbp = str(tmp_path / "body.db")
    assert main_import([BODY_FAILS, "--db", dbp]) == 0

    failed = _failed_hands(dbp)
    assert len(failed) == 1, "the bad hand must not take its neighbour down"
    label = failed[0]["label"]
    assert BODY_FAILS_UID in label, (
        f"hand #{BODY_FAILS_UID} parsed its header and is named in the stored "
        f"raw text, but the report labelled it {label!r}")
    assert label != UNIDENTIFIED_HAND


def test_header_failure_stays_unidentified(tmp_path) -> None:
    """No hand number was ever read, so none may be reported."""
    dbp = str(tmp_path / "header.db")
    assert main_import([HEADER_FAILS, "--db", dbp]) == 0

    failed = _failed_hands(dbp)
    assert len(failed) == 1
    assert failed[0]["label"] == UNIDENTIFIED_HAND


def test_the_stored_raw_carries_the_number_the_report_dropped(tmp_path) -> None:
    """Locates the loss: the uid is present in `raw` but absent from the column.

    Kept separate from the rendering test because it distinguishes two very
    different repairs — teaching the REPORT to dig a number out of the raw
    text, versus recording it at the point where it was already parsed. Only
    the second is right, and this is the assertion that says so.
    """
    dbp = str(tmp_path / "raw.db")
    assert main_import([BODY_FAILS, "--db", dbp]) == 0

    conn = db.connect(dbp)
    rows = db.failed_hands(conn)
    assert len(rows) == 1
    assert BODY_FAILS_UID in rows[0]["raw"], "fixture must carry a real uid"
    assert rows[0]["hand_uid"] == BODY_FAILS_UID
