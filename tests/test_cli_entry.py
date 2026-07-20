"""Round-3 finding [P4'] (P1): the pipeline had no entry points.

Importing hands and draining the solve backlog were reachable only by writing
Python against `hh.persist` — the two operations the training loop is *made of*
(plan §5.3) had no way to be run. These are thin argparse wrappers over the
existing functions; the tests pin the seam (files in -> session persisted, queue
drained) and the counts the operator has to see, not new logic.
"""

from pathlib import Path

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
