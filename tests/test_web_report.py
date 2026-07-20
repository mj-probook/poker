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

from pokerlab.drills import generator as gen
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
    # terminal rows are reported by CAUSE, not as one "failed" number — the
    # three causes have three different remedies (see the terminal-cause tests)
    assert "blocked" in s


def test_report_ranks_the_exact_tier_leaks_by_ev_loss(imported):
    # `leaks` is {rows, approx_caveat} since [R4-1] — the ranking and the
    # disclosure its numbers require travel together, like tier3.
    leaks = _report(imported)["leaks"]["rows"]
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
    ranked = {r["leak_key"] for r in body["leaks"]["rows"]}
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


def test_the_tier_1_claim_on_every_drill_is_warranted_by_its_solution():
    """`_spot_json` asserts tier 1 for EVERY drill — pin that it is entitled to.

    The comment beside that line says the tier is "stated, not assumed, because
    a future solver-backed drill kind must not silently inherit a
    'chart-graded' claim". It described a protection that did not exist: the
    value was hardcoded and nothing checked the drill was chart-sourced, so the
    claim held for all 5408 drills on the strength of a 1-drill pin elsewhere.
    A comment asserting an enforcement is worse than no comment — it stops the
    next reader re-deriving it (r2-edge-hunter). Fourth instance of the
    recorded-vs-landed pattern, this time as a code comment.

    This is the enforcement. Direct analogue of the EV_UNITS kind pin: the
    served tier label is only honest while every drill really is answered by
    the in-house chart engine, so the first non-chart drill kind fails HERE
    rather than shipping a false "exact — chart-graded" badge on a load-bearing
    UI surface (plan §9).
    """
    sources = {d.solution.source for d in gen.default_population()}
    assert sources == {"chart"}, (
        f"_spot_json claims TIER_CHART for every drill, but the population "
        f"carries sources {sorted(sources)} — the tier label is no longer "
        f"warranted and must be derived per drill, not hardcoded")


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


# --------------------------------------------------------------------------- #
# [P5'] follow-on: a terminal row is not "backlog". Before r1's c191f51 every
# terminal cause was status='failed', so the report could only say a number —
# and the PARTIAL banner promised "run pokerlab-batch and these numbers will
# change" forever, because gate-refused rows never left the queue. The three
# causes have three DIFFERENT remedies, and a report that does not say which
# one applies is telling the operator to do the wrong thing.
# --------------------------------------------------------------------------- #
def _queued(conn, spot_key, idx, status):
    hid = db.insert_imported_hand(conn, "PokerStars", "raw", "{}", AT, f"u{idx}")
    rid = db.enqueue_batch(conn, spot_key, hid, idx)
    if status != "pending":
        db.set_batch_status(conn, rid, status)


def test_terminal_causes_are_reported_apart_each_with_its_own_remedy(tmp_path):
    dbp = str(tmp_path / "t.db")
    conn = db.connect(dbp)
    _queued(conn, "a|flop", 0, "unsolvable")
    _queued(conn, "b|flop", 1, "mismatched")
    _queued(conn, "c|flop", 2, "failed")
    conn.close()

    blocked = {b["status"]: b for b in _report(dbp)["session"]["blocked"]}

    assert set(blocked) == {"unsolvable", "mismatched", "failed"}
    assert all(b["count"] == 1 for b in blocked.values())
    # each carries a DISTINCT operator action — that is the whole point
    actions = {b["status"]: b["action"] for b in blocked.values()}
    assert len(set(actions.values())) == 3
    assert "solver" in actions["unsolvable"].lower()
    assert "import" in actions["mismatched"].lower()   # NOT "retry" — it re-fails
    assert "retry" in actions["failed"].lower()


def test_partial_means_what_it_says_once_nothing_is_pending(tmp_path):
    """The original defect: terminal rows kept the session PARTIAL forever."""
    dbp = str(tmp_path / "t.db")
    conn = db.connect(dbp)
    _queued(conn, "a|flop", 0, "unsolvable")
    conn.close()

    s = _report(dbp)["session"]
    assert s["queued"] == 0            # an unsolvable row is not pending work
    assert s["partial"] is False       # ...so the report is COMPLETE
    assert s["blocked"][0]["count"] == 1   # but the row is still visible


def test_a_pending_row_still_makes_the_session_partial(tmp_path):
    dbp = str(tmp_path / "t.db")
    conn = db.connect(dbp)
    _queued(conn, "a|flop", 0, "pending")
    conn.close()

    s = _report(dbp)["session"]
    assert s["queued"] == 1 and s["partial"] is True


def test_report_copy_and_terminal_statuses_agree_in_both_directions(tmp_path):
    """The store owns the vocabulary; this module owns only the copy.

    Compared against db.TERMINAL_STATUSES rather than restating the list, and
    asserted BOTH ways so each failure names its own cause:

      * a terminal status with no copy would render as a bare count with no
        remedy — the defect this whole surface exists to fix, reappearing
        silently the moment someone adds a fourth cause;
      * copy for something that is not a terminal status is dead text that
        `_report` can never emit, and it would quietly outlive a status the
        store had renamed or dropped.
    """
    from pokerlab.web.app import TERMINAL_ACTIONS

    statuses, copy = set(db.TERMINAL_STATUSES), set(TERMINAL_ACTIONS)
    assert not (statuses - copy), f"terminal statuses with no operator copy: {statuses - copy}"
    assert not (copy - statuses), f"operator copy for non-statuses: {copy - statuses}"
    # and every terminal status is a real member of the batch vocabulary
    assert statuses <= set(db.BATCH_STATUSES)


def test_report_js_renders_each_terminal_cause_with_its_served_action():
    """Same rule as the tier label: the page renders copy it does not author."""
    js = (STATIC / "report.js").read_text()
    from pokerlab.web.app import TERMINAL_ACTIONS

    assert "s.blocked" in js
    assert "b.action" in js and "b.status" in js
    # The page must not RESTATE the server's wording. Banning the individual
    # words would also ban the comments that explain the design, so the check
    # is against the actual copy: no TERMINAL_ACTIONS string may appear here.
    for action in TERMINAL_ACTIONS.values():
        assert action not in js, f"copy must come from the payload: {action!r}"


# --------------------------------------------------------------------------- #
# Sweep finding [R1] (P2): PLAN §8 M4 exit promises failed hands are "isolated
# and surfaced in the session summary — never silently dropped". r1's d2704b3
# made the failure durable; this is the surface. A count alone would recreate
# the not-actionable defect the terminal-cause split just fixed, so each failure
# carries its identity and its reason, and the section carries its remedy.
# --------------------------------------------------------------------------- #
@pytest.fixture
def one_good_one_broken(tmp_path):
    """r3-product's acceptance fixture, built to actually discriminate.

    The GOOD hand is `ps_multiway_flop.txt` (5 gradings), never
    `ps_preflop_fold.txt` — that one parses fine but grades nothing, so "the
    good hand still landed" would assert nothing at all (r1's fixture note).
    """
    dbp = str(tmp_path / "mixed.db")
    good = (FIXTURES / "ps_multiway_flop.txt").read_text()
    broken = "PokerStars Hand #999: this header parses, the body does not\n"
    parsed = [parse_pokerstars(good)]
    conn = db.connect(dbp)
    report = grade_session(parsed, population=load_population())
    persist_session(conn, parsed, report, graded_at=AT, raw_texts=[good])
    # the broken hand's failure is what r1's table records
    db.insert_failed_hand(conn, "PokerStars", broken,
                          "ValueError: unparseable body", AT, hand_uid="999")
    # ...and one whose parse died before it reached a hand number
    db.insert_failed_hand(conn, "GGPoker", "garbage", "ValueError: no header",
                          AT, hand_uid=None)
    conn.close()
    return dbp


def test_failed_hands_are_surfaced_never_silently_dropped(one_good_one_broken):
    s = _report(one_good_one_broken)["session"]

    assert s["failed"] == 2
    # a count alone is not actionable: each failure names itself and its cause
    reasons = {f["reason"] for f in s["failed_hands"]}
    assert "ValueError: unparseable body" in reasons
    assert all("site" in f and "reason" in f for f in s["failed_hands"])
    # and the section states the remedy, like the terminal batch causes
    assert "failed_action" in s and "import" in s["failed_action"].lower()


def test_a_failure_with_no_hand_number_renders_unidentified(one_good_one_broken):
    """`hand_uid` is nullable — the parse can die before reaching it.

    A missing identifier is UNKNOWN, not absent: same honesty rule as a NULL
    population_frequency rendering "no baseline" rather than 0%. Never a blank,
    never a fabricated id.
    """
    by_site = {f["site"]: f for f in
               _report(one_good_one_broken)["session"]["failed_hands"]}

    assert by_site["PokerStars"]["hand_uid"] == "999"
    assert by_site["GGPoker"]["hand_uid"] is None      # served as null...
    assert by_site["GGPoker"]["label"] == "unidentified"   # ...rendered as this
    assert by_site["PokerStars"]["label"] == "999"


def test_the_good_hand_still_landed_alongside_the_failures(one_good_one_broken):
    """One broken hand must never cost the hands that did import (finding [8])."""
    body = _report(one_good_one_broken)
    assert body["session"]["hands"] == 1
    assert body["session"]["graded"] > 0
    assert body["leaks"]["rows"] or body["tier3"]["rows"], "the good hand produced output"


def test_partial_means_batch_pending_only_not_failure(one_good_one_broken):
    """THE DELIBERATE PIN (sweep ruling). `partial` answers exactly one
    question: "will draining change these numbers?"

    A failed hand's numbers will NOT change from draining — its remedy is a
    parser fix and a re-import, a different action entirely. Widening `partial`
    to mean "incomplete in any sense" would restore the defect fixed in
    79f0ff5, where it was permanently true under a banner promising a drain
    would move numbers it could never move.

    The cost is real and is why the two assertions live in ONE test: a session
    can read `partial: false` while carrying ungraded hands. That is only
    honest because the failure surface is unmissable, so the `false` and the
    visible failures are pinned TOGETHER — neither may be changed alone.
    """
    # Resolve the good hand's tier-2 backlog first — that is the ONLY thing
    # `partial` is allowed to track, so it has to be empty for this test to be
    # about failures at all. (Terminal, as a real drain leaves gate-refused
    # spots; the point is that nothing is pending.)
    conn = db.connect(one_good_one_broken)
    for row in db.pending_batch(conn):
        db.set_batch_status(conn, row["id"], "unsolvable")
    conn.close()

    s = _report(one_good_one_broken)["session"]

    assert s["queued"] == 0
    assert s["partial"] is False          # nothing to drain -> not partial...
    assert s["failed"] == 2               # ...and the failures are RIGHT THERE
    assert len(s["failed_hands"]) == 2


def test_skipped_and_failed_are_never_collapsed(tmp_path):
    """Dedup-skipped and failed are different events with different meanings.

    "Already imported, nothing to do" versus "you lost a hand". Collapsing
    remedies at the action layer is the defect caught in retry_failed_batch
    (r1's note), and it would be the same defect here.
    """
    dbp = str(tmp_path / "dup.db")
    raw = (FIXTURES / "ps_multiway_flop.txt").read_text()
    parsed = [parse_pokerstars(raw)]
    conn = db.connect(dbp)
    rep = grade_session(parsed, population=load_population())
    persist_session(conn, parsed, rep, graded_at=AT, raw_texts=[raw])
    second = persist_session(conn, parsed, rep, graded_at=AT, raw_texts=[raw])
    conn.close()

    assert second["skipped"] == 1 and second["failed"] == 0
    s = _report(dbp)["session"]
    assert s["failed"] == 0, "a dedup skip is not a failure"


def test_report_js_renders_failed_hands_with_the_served_action_and_label():
    """Same rule as every other claim on this page: rendered, never authored."""
    from pokerlab.web.app import FAILED_HANDS_ACTION, UNIDENTIFIED_HAND

    js = (STATIC / "report.js").read_text()
    assert "s.failed_hands" in js and "s.failed_action" in js
    assert "f.reason" in js and "f.label" in js
    # the remedy copy and the unknown-id wording both live server-side
    assert FAILED_HANDS_ACTION not in js
    assert UNIDENTIFIED_HAND not in js, "the page must not author the id fallback"
