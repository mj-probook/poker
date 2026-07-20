"""Slice F — multi-hand files (round-4 finding [R2']).

`parse_pokerstars`/`parse_ggpoker` read ONE hand per file, but plan §5.3 commits
to "PokerStars (auto-saved local files)", which hold hundreds of hands
concatenated. Fed a real file, the parser matched the header on line 0, ran the
body grammar over EVERY line including subsequent hands' lines, and returned a
single ParsedHand carrying the first hand's id with every hand's actions merged
into it — silently, no exception. The pipeline only caught it one stage later,
where `extract_decisions` desynced; both hands were lost to report one error.

Three review rounds missed this because all 12 fixtures are single-hand. They
were built to match the parser's assumption, so the suite confirmed the
assumption instead of testing it. Multi-hand is a PERMANENT fixture axis for
that reason, like the short-stack precedent.
"""

from pathlib import Path

import pytest

from pokerlab.hh.decisions import extract_decisions
from pokerlab.hh.ggpoker import parse_ggpoker
from pokerlab.hh.pokerstars import parse_pokerstars

FIXTURES = Path(__file__).parent / "fixtures" / "hh"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text()


def test_pokerstars_session_file_yields_every_hand():
    """The headline invariant: N headers in, N hands out."""
    from pokerlab.hh.pokerstars import parse_pokerstars_file

    raw = _read("ps_session_multi.txt")
    hands = parse_pokerstars_file(raw)
    assert len(hands) == 3, f"3 concatenated hands -> {len(hands)}"
    # distinct hands, not the same one repeated
    assert len({h.hand_id for h in hands}) == 3


def test_ggpoker_session_file_yields_every_hand():
    """GGPoker shares `_common.parse_hand`, so it shared the defect."""
    from pokerlab.hh.ggpoker import parse_ggpoker_file

    hands = parse_ggpoker_file(_read("gg_session_multi.txt"))
    assert len(hands) == 2
    assert len({h.hand_id for h in hands}) == 2


def test_each_hand_matches_its_standalone_parse():
    """Splitting must not perturb the hands — the merge symptom's direct inverse.

    Pre-fix, a 2-hand file produced ONE hand carrying 21 actions (6 from the
    first, 15 from the second). Asserting per-hand equality against the
    standalone parse is what pins that the actions land on the right hand,
    rather than merely that the right NUMBER of hands came back.
    """
    from pokerlab.hh.pokerstars import parse_pokerstars_file

    standalone = [parse_pokerstars(_read(n)) for n in
                  ("ps_preflop_fold.txt", "ps_multiway_flop.txt",
                   "ps_allin_sidepot.txt")]
    split = parse_pokerstars_file(_read("ps_session_multi.txt"))

    assert [h.hand_id for h in split] == [h.hand_id for h in standalone]
    for got, want in zip(split, standalone):
        assert len(got.actions) == len(want.actions), (
            f"hand {got.hand_id}: {len(got.actions)} actions vs "
            f"{len(want.actions)} parsed standalone")
        assert got.actions == want.actions
        # setup + seat identity + settlement: a split that got the action count
        # right while mis-slicing the SUMMARY would still corrupt the hand.
        assert got.seat_names == want.seat_names
        assert got.setup == want.setup
        assert got.hero == want.hero
        assert got.contributed == want.contributed
        assert got.collected == want.collected
        assert got.total_pot == want.total_pot


def test_every_hand_in_the_file_is_gradeable():
    """End-to-end: the stage that actually broke was one past the parser.

    `extract_decisions` is where the merged hand raised its desync, so the
    invariant is not just "N hands parsed" but "N hands each survive the next
    stage" — the pre-fix file produced zero usable decisions from two valid
    hands.
    """
    from pokerlab.hh.pokerstars import parse_pokerstars_file

    hands = parse_pokerstars_file(_read("ps_session_multi.txt"))
    per_hand = [extract_decisions(h) for h in hands]
    assert all(d for d in per_hand), "a hand yielded no decisions"


def test_single_hand_file_still_returns_one_hand():
    """The plural path must not require plurality — most fixtures are single."""
    from pokerlab.hh.pokerstars import parse_pokerstars_file

    hands = parse_pokerstars_file(_read("ps_preflop_fold.txt"))
    assert len(hands) == 1
    assert hands[0].hand_id == parse_pokerstars(_read("ps_preflop_fold.txt")).hand_id


def test_singular_parser_refuses_a_multi_hand_file():
    """The silent merge itself is fixed, not merely bypassed.

    The plural entry point could be added while `parse_pokerstars` kept
    silently merging, leaving the trap armed for every existing caller. A
    singular parser handed plural input must fail loudly instead of inventing a
    hand that never existed.
    """
    with pytest.raises(ValueError, match="2 hands"):
        parse_pokerstars(_read("ps_session_multi.txt")[:_read(
            "ps_session_multi.txt").index("PokerStars Hand #", 10)]
            + _read("ps_preflop_fold.txt"))


def test_ggpoker_singular_parser_also_refuses():
    two = _read("gg_session_multi.txt")
    with pytest.raises(ValueError, match="hands"):
        parse_ggpoker(two)


# --------------------------------------------------------------------------- #
# The CLI import path. Cutting the split at the call site left the ENTIRE suite
# green while real imports silently merged every file again — the parser-level
# tests above all pass through `parse_pokerstars_file`, which is not the
# function `pokerlab-import` calls. Splitting and per-hand isolation are
# separate wires and the second one had no test at all (the round-3 [P7']
# lesson: a revert-check proves the wire you cut and nothing about the others).
# --------------------------------------------------------------------------- #
def test_cli_import_path_splits_session_files():
    from pokerlab.cli import _read_hands

    parsed, raws, problems, lost = _read_hands([str(FIXTURES / "ps_session_multi.txt")])
    assert len(parsed) == 3, f"CLI parsed {len(parsed)} hands from a 3-hand file"
    assert not problems and not lost
    # each hand's raw is its OWN chunk, not the whole file: `raws` is the
    # per-hand re-import payload, so storing the session under every hand would
    # make one bad hand un-re-importable without re-importing all of them.
    assert len(raws) == 3
    assert all(r.count("PokerStars Hand #") == 1 for r in raws)


def test_one_bad_hand_does_not_cost_the_good_ones():
    """Isolation moved from per-FILE to per-HAND — that is the whole point.

    Pre-fix this file yielded zero usable hands: the broken chunk in the middle
    took the two valid ones down with it, because the unit of isolation was the
    file.
    """
    from pokerlab.cli import _read_hands

    parsed, raws, problems, lost = _read_hands([str(FIXTURES / "ps_session_mixed.txt")])
    assert len(parsed) == 2, "the two valid hands must survive the broken one"
    assert len(lost) == 1, "the broken hand must be recorded, not dropped"
    site, raw, reason = lost[0]
    assert site == "PokerStars"
    assert "BROKEN" in raw, "the lost chunk's OWN text is stored for re-import"
    assert raw.count("PokerStars Hand #") == 1, "lost raw is one hand, not the file"
    # the message names which hand in the file, not just the filename
    assert any("hand 2/3" in p for p in problems), problems


def test_header_failure_reaches_failed_hands_durably(tmp_path):
    """[P2] A header-level failure used to die on stdout and leave no trace.

    Also the first exercise of the nullable `hand_uid` path: the parse never
    got far enough to learn a hand number, so the row is honestly site-only.
    """
    import sqlite3

    from pokerlab.cli import main_import

    dbp = str(tmp_path / "import.db")
    main_import([str(FIXTURES / "ps_session_mixed.txt"), "--db", dbp])
    con = sqlite3.connect(dbp)
    rows = con.execute("SELECT site, hand_uid, raw FROM failed_hands").fetchall()
    assert len(rows) == 1, f"expected 1 failed_hands row, got {len(rows)}"
    site, uid, raw = rows[0]
    assert site == "PokerStars"
    assert uid is None, "no hand number was parseable — the row must say so"
    assert "BROKEN" in raw
    # and the good hands still landed
    assert con.execute("SELECT COUNT(*) FROM imported_hands").fetchone()[0] == 2


def test_unrecognized_file_is_still_not_a_lost_hand():
    """Splitting must not turn 'not a hand history' into a fabricated failure.

    `failed_hands.site` is NOT NULL, and the honest reason no invented "unknown"
    site is needed is that an unattributable file never reaches the per-hand
    path at all. That boundary is easy to erode when adding a splitter, so it
    is pinned.
    """
    from pokerlab.cli import _read_hands

    junk = FIXTURES / "ps_not_a_hand_history.txt"
    junk.write_text("not a hand history at all\n")
    try:
        parsed, raws, problems, lost = _read_hands([str(junk)])
        assert not parsed and not lost
        assert any("unrecognized" in p for p in problems)
    finally:
        junk.unlink()


def test_main_import_actually_calls_the_splitter(tmp_path):
    """The SEAM: `main_import` -> splitter. Owned by neither suite until now.

    The splitter tests above call `split_pokerstars`/`parse_pokerstars_file`
    directly; `test_cli_entry.py`'s fixtures are single-hand, for which
    `[raw]` and `split(raw)` are identical. So both suites stayed green with the
    CLI wiring absent — imports, a `_SPLITTERS` table and a five-line comment
    all asserting per-hand splitting while the enforcement line did nothing.

    That is the wave's inertness pattern in the seam between two owners:
    everything each side owns is tested, and the wire between them is not. This
    routes a real multi-hand file through the actual entry point and asserts the
    hand count, which is the only assertion the inert version fails.
    """
    import sqlite3

    from pokerlab.cli import main_import

    dbp = str(tmp_path / "seam.db")
    main_import([str(FIXTURES / "ps_session_multi.txt"), "--db", dbp])
    n = sqlite3.connect(dbp).execute(
        "SELECT COUNT(*) FROM imported_hands").fetchone()[0]
    assert n > 1, (
        f"main_import stored {n} hand(s) from a 3-hand session file — "
        "the CLI is not calling the splitter")
    assert n == 3


# --------------------------------------------------------------------------- #
# Chunk byte-faithfulness. `failed_hands` is keyed on the stored raw text and
# `clear_failed_hand` matches it exactly, so once `raw` became a CHUNK the
# clear's correctness started depending on this splitter. The first version
# joined `splitlines()` with "\n" — silently normalizing CRLF and dropping the
# trailing newline — so no chunk was byte-identical to its source and every
# failure recorded before the change became permanently unclearable, with the
# report claiming forever that a fixed hand is broken.
#
# Making the chunker byte-faithful dissolves that seam rather than pinning a
# boundary convention or re-keying the store: the property promised is
# "concatenates back to the input", which is worth guaranteeing on its own
# terms and lets the splitter change freely as long as it holds.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["ps_preflop_fold.txt", "ps_session_multi.txt",
                                  "ps_session_mixed.txt"])
def test_chunks_concatenate_back_to_the_source_exactly(name):
    from pokerlab.hh.pokerstars import split_pokerstars

    raw = _read(name)
    assert "".join(split_pokerstars(raw)) == raw


def test_windows_line_endings_survive_the_split():
    """A PokerStars client on Windows writes CRLF; normalizing it loses the key."""
    from pokerlab.hh.pokerstars import split_pokerstars

    crlf = _read("ps_session_multi.txt").replace("\n", "\r\n")
    assert "".join(split_pokerstars(crlf)) == crlf


def test_a_failure_recorded_before_splitting_still_clears(tmp_path):
    """The end-to-end consequence: fail a hand, fix it, re-import, count goes to 0.

    Pinned end-to-end rather than as a string property because the property is
    only interesting for what it protects. A row written under the pre-split
    shape (raw = the whole single-hand file, exactly as R1 wrote it) must still
    be cleared by a successful re-import; otherwise `failed_hands` answers "what
    was ever broken" instead of "what is broken now".
    """
    import sqlite3

    from pokerlab.cli import main_import
    from pokerlab.store import db

    dbp = str(tmp_path / "legacy.db")
    raw = _read("ps_preflop_fold.txt")
    conn = db.connect(dbp)
    db.insert_failed_hand(conn, "PokerStars", raw, "ValueError: old parser bug",
                          "2026-07-19T00:00:00")
    conn.close()
    assert sqlite3.connect(dbp).execute(
        "SELECT COUNT(*) FROM failed_hands").fetchone()[0] == 1

    main_import([str(FIXTURES / "ps_preflop_fold.txt"), "--db", dbp])
    remaining = sqlite3.connect(dbp).execute(
        "SELECT COUNT(*) FROM failed_hands").fetchone()[0]
    assert remaining == 0, (
        "a pre-split failure row survived a successful re-import — the report "
        "will claim forever that this hand is broken")
