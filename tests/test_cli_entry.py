"""Round-3 finding [P4'] (P1): the pipeline had no entry points.

Importing hands and draining the solve backlog were reachable only by writing
Python against `hh.persist` — the two operations the training loop is *made of*
(plan §5.3) had no way to be run. These are thin argparse wrappers over the
existing functions; the tests pin the seam (files in -> session persisted, queue
drained) and the counts the operator has to see, not new logic.
"""

from pathlib import Path

import pytest

from pokerlab.cli import main_batch, main_import
from pokerlab.spike import tier2
from pokerlab.store import db

FIXTURES = Path(__file__).parent / "fixtures" / "hh"
_FILES = [str(FIXTURES / f) for f in
          ("ps_multiway_flop.txt", "ps_ante_hu.txt", "gg_sb_fold_leak.txt")]


def test_import_parses_grades_and_persists_a_session(tmp_path, capsys) -> None:
    dbp = str(tmp_path / "cli.db")
    assert main_import([*_FILES, "--db", dbp]) == 0

    out = capsys.readouterr().out
    assert "imported" in out and "graded" in out
    # the operator must be able to see BOTH honesty signals from the summary
    assert "failed" in out and "pending" in out

    conn = db.connect(dbp)
    assert len(db.gradings(conn)) > 0
    # and the import wired its leaks into the drill scheduler (P1')
    assert db.all_sr_state(conn), "import must seed drillable categories"


def test_import_routes_each_file_to_its_sites_parser(tmp_path) -> None:
    """Both v1 sites in one invocation (plan §5.3), keyed off the HH header."""
    dbp = str(tmp_path / "cli.db")
    assert main_import([*_FILES, "--db", dbp]) == 0
    sites = {h["site"] for h in
             db.connect(dbp).execute("SELECT site FROM imported_hands")}
    assert sites == {"PokerStars", "GGPoker"}


def test_import_reports_an_unparseable_file_without_losing_the_session(
        tmp_path, capsys) -> None:
    """One bad file must never cost the hands that did import (finding [8])."""
    bad = tmp_path / "junk.txt"
    bad.write_text("not a hand history at all\n")
    dbp = str(tmp_path / "cli.db")

    assert main_import([*_FILES, str(bad), "--db", dbp]) == 0

    assert "junk.txt" in capsys.readouterr().out
    assert len(db.gradings(db.connect(dbp))) > 0


def test_batch_drains_once_and_prints_every_outcome(tmp_path, capsys,
                                                    monkeypatch) -> None:
    """The wrapper's own seam: argv -> drain -> every outcome on the console.

    The solver is stubbed to a miss so this pins the CLI, not the solve path —
    a real solve here would make the test a (slow) assertion about tier-2
    behaviour that `test_hh_store` already owns.
    """
    dbp = str(tmp_path / "cli.db")
    main_import([*_FILES, "--db", dbp])
    capsys.readouterr()
    monkeypatch.setattr(tier2, "make_drain_solver",
                        lambda conn, **kw: lambda spot_key, d: None)

    assert main_batch(["--db", dbp, "--iters", "1"]) == 0

    out = capsys.readouterr().out
    # a miss and an unsolvable spot are DIFFERENT outcomes (wave-2 [E15]) and
    # the operator has to be able to tell them apart from the console
    for outcome in ("done", "missed", "unsolvable", "failed", "recovered"):
        assert outcome in out


def test_batch_on_an_empty_queue_is_a_no_op(tmp_path, capsys) -> None:
    dbp = str(tmp_path / "empty.db")
    db.connect(dbp)
    assert main_batch(["--db", dbp]) == 0
    assert "done" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# [P5' follow-on] The report tells the operator "fix the bug, then retry the
# batch" for a failed row. That advice needs a way to ACT on it: the drain only
# picks up 'pending', so without this the only route was writing Python against
# db.retry_failed_batch — the exact gap P4' exists to close.
# --------------------------------------------------------------------------- #
def _one_failed_row(dbp):
    conn = db.connect(dbp)
    hid = db.insert_imported_hand(conn, "PokerStars", "raw", "{}", "T", "u1")
    rid = db.enqueue_batch(conn, "a|flop", hid, 0)
    db.set_batch_status(conn, rid, "failed")
    conn.commit()
    return conn


def test_batch_retry_reopens_bug_failures(tmp_path, capsys, monkeypatch):
    dbp = str(tmp_path / "r.db")
    _one_failed_row(dbp)
    monkeypatch.setattr(tier2, "make_drain_solver",
                        lambda conn, **kw: lambda spot_key, d: None)

    assert main_batch(["--db", dbp, "--retry"]) == 0

    # The contract of --retry is that the row is REOPENED before the drain.
    # Asserting its status afterwards would test something else entirely: this
    # fixture's raw text is not a real hand history, so the drain re-derives it,
    # fails, and terminates the row again — correct behaviour that has nothing
    # to do with whether the reopen worked.
    assert "reopened    : 1" in capsys.readouterr().out


def test_batch_retry_leaves_unsolvable_alone_by_default(tmp_path, monkeypatch):
    """Retrying an unsolvable row re-fails by construction — so don't.

    Mirrors db.retry_failed_batch's default. The CLI must not quietly widen it,
    or the tooling starts recommending the loop the report warns against.
    """
    dbp = str(tmp_path / "r.db")
    conn = db.connect(dbp)
    hid = db.insert_imported_hand(conn, "PokerStars", "raw", "{}", "T", "u1")
    for i, st in enumerate(("failed", "unsolvable", "mismatched")):
        rid = db.enqueue_batch(conn, f"s{i}|flop", hid, i)
        db.set_batch_status(conn, rid, st)
    conn.commit()
    monkeypatch.setattr(tier2, "make_drain_solver",
                        lambda conn, **kw: lambda spot_key, d: None)

    main_batch(["--db", dbp, "--retry"])

    left = {r["spot_key"]: r["status"] for r in db.batch_rows(db.connect(dbp))}
    assert left["s1|flop"] == "unsolvable"   # needs a solver upgrade, not a retry
    assert left["s2|flop"] == "mismatched"   # needs a re-import, not a retry


def test_batch_retry_accepts_an_explicit_status(tmp_path, capsys, monkeypatch):
    """After a solver upgrade the operator can reopen that class deliberately."""
    dbp = str(tmp_path / "r.db")
    conn = db.connect(dbp)
    hid = db.insert_imported_hand(conn, "PokerStars", "raw", "{}", "T", "u1")
    rid = db.enqueue_batch(conn, "a|flop", hid, 0)
    db.set_batch_status(conn, rid, "unsolvable")
    conn.commit()
    monkeypatch.setattr(tier2, "make_drain_solver",
                        lambda conn, **kw: lambda spot_key, d: None)

    main_batch(["--db", dbp, "--retry", "unsolvable"])

    assert "reopened    : 1 (unsolvable)" in capsys.readouterr().out


def test_batch_rejects_a_non_terminal_retry_status(tmp_path, capsys) -> None:
    """A bad status is a USAGE error — argparse's exit 2, not a silent no-op."""
    dbp = str(tmp_path / "r.db")
    db.connect(dbp)
    with pytest.raises(SystemExit) as exc:
        main_batch(["--db", dbp, "--retry", "done"])
    assert exc.value.code == 2
    assert "terminal" in capsys.readouterr().err.lower()
