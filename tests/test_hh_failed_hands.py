"""Wave-3 sweep [R1]: a failed hand must leave a DURABLE trace.

PLAN §8 M4 promises hands failing parse/replay are "isolated in `failed_hands`
and surfaced in the session summary — never silently dropped". Round-1 [8]
delivered the isolation as an in-memory dataclass on `SessionReport`, which is
correct for the CLI and vanishes with the process. On the intended nightly-cron
path nobody reads stdout, so by morning a failed hand had left no trace at all:
`/api/report` showed the session as complete and short by one hand, with no
failure count anywhere.

This covers the STORE half. The report surface is w3-product-builder's.
"""

import sqlite3
from pathlib import Path

import pytest

from pokerlab.hh.persist import failed_hand_report, persist_session
from pokerlab.hh.pokerstars import parse_pokerstars
from pokerlab.hh.population import load_population
from pokerlab.hh.report import FailedHand, grade_session
from pokerlab.store import db

FIXTURES = Path(__file__).parent / "fixtures" / "hh"
AT = "2026-07-19T00:00:00"
LATER = "2026-07-20T00:00:00"


def _pair():
    """Two DISTINCT hands — same-hand-twice would dedup and mask the result."""
    out = []
    # good hand first. It must be one whose decisions actually grade -- the
    # preflop fold parses fine but its only decision is graded=False, so using
    # it as the good hand would assert nothing about gradings surviving.
    for name in ("ps_multiway_flop.txt", "ps_preflop_fold.txt"):
        raw = (FIXTURES / name).read_text()
        out.append((raw, parse_pokerstars(raw)))
    return out


def _session_with_one_broken_hand(conn, *, graded_at=AT, break_it=True):
    """r3-product's acceptance shape: one good hand, one that cannot be graded.

    The break is injected at the report level rather than by shipping a corrupt
    fixture, so the test names exactly which hand failed and why, without
    depending on some particular parser bug staying unfixed to stay meaningful.
    """
    pair = _pair()
    hands = [p for _, p in pair]
    raws = [pair[0][0], pair[1][0]]
    report = grade_session(hands, population=load_population())
    if break_it:
        report.graded[:] = [g for g in report.graded if g.hand_index != 1]
        report.failed_hands.append(
            FailedHand(1, hands[1].hand_id, "ValueError: unparseable street"))
    return persist_session(conn, hands, report, graded_at=graded_at,
                           raw_texts=raws)


def test_a_failed_hand_is_recorded_not_dropped():
    conn = db.connect()
    counts = _session_with_one_broken_hand(conn)

    assert counts["failed"] == 1
    rows = failed_hand_report(conn)
    assert len(rows) == 1
    assert rows[0]["reason"] == "ValueError: unparseable street"
    assert rows[0]["imported_at"] == AT


def test_the_good_hand_still_lands():
    """One bad hand must not cost the session — the [8] invariant, persisted."""
    conn = db.connect()
    counts = _session_with_one_broken_hand(conn)

    assert counts["imported"] == 1 and counts["graded"] > 0
    assert len(db.gradings(conn)) == counts["graded"]


def test_the_raw_text_is_kept_so_the_hand_can_be_reimported():
    """Storing the text IS the re-import path; without it the record is a stub."""
    conn = db.connect()
    _session_with_one_broken_hand(conn)
    expected = (FIXTURES / "ps_preflop_fold.txt").read_text()
    assert failed_hand_report(conn)[0]["raw"] == expected


def test_a_failed_hand_does_not_claim_its_dedup_uid():
    """[E16]/[E17] must survive: recording the failure must not make it permanent.

    If the failed hand claimed (site, hand_uid) in `imported_hands`, a later
    re-import after the parser fix would be skipped as a duplicate and the hand
    would be lost forever — the failure record would have caused the loss it
    exists to prevent.
    """
    conn = db.connect()
    _session_with_one_broken_hand(conn)

    uid = failed_hand_report(conn)[0]["hand_uid"]
    assert uid, "the parse got far enough to know the hand number"
    assert not conn.execute(
        "SELECT 1 FROM imported_hands WHERE hand_uid=?", (uid,)).fetchone()


def test_reimporting_after_a_fix_clears_the_failure():
    """Otherwise the summary reports a hand that now grades fine, forever."""
    conn = db.connect()
    _session_with_one_broken_hand(conn)
    assert len(failed_hand_report(conn)) == 1

    # the parser is fixed: the same session re-imports, nothing fails
    _session_with_one_broken_hand(conn, graded_at=LATER, break_it=False)

    assert failed_hand_report(conn) == [], (
        "a fixed hand must stop being reported as failed")


def test_failures_are_scoped_to_an_import_batch():
    conn = db.connect()
    _session_with_one_broken_hand(conn)
    assert len(failed_hand_report(conn, AT)) == 1
    assert failed_hand_report(conn, LATER) == []


def test_the_record_is_written_in_the_import_transaction():
    """A rollback must take the failure record with it.

    `persist_session` is one transaction so a session lands whole or not at all
    (round-1 [11]). A failure row committed outside it would survive a rolled-
    back import and report a failure for a session that never landed.
    """
    conn = db.connect()
    pair = _pair()
    hands = [p for _, p in pair]
    report = grade_session(hands, population=load_population())
    report.graded[:] = [g for g in report.graded if g.hand_index != 0]
    report.failed_hands.append(FailedHand(0, hands[0].hand_id, "boom"))

    # hand 0 is written to failed_hands, then hand 1 raises before the commit
    with pytest.raises(IndexError):
        persist_session(conn, hands, report, graded_at=AT,
                        raw_texts=[pair[0][0]])   # one raw for two hands

    assert failed_hand_report(conn) == [], "the failure row outlived its rollback"


# --------------------------------------------------------------------------- #
# A database created before this table existed must gain it on connect.
# --------------------------------------------------------------------------- #
def test_a_legacy_db_gains_the_failed_hands_table(tmp_path):
    p = tmp_path / "old.db"
    c = sqlite3.connect(str(p))
    c.executescript("""
        CREATE TABLE imported_hands(id INTEGER PRIMARY KEY, site TEXT NOT NULL,
          raw TEXT NOT NULL, parsed_json TEXT NOT NULL, imported_at TEXT NOT NULL,
          hand_uid TEXT, UNIQUE(site, hand_uid));
    """)
    c.commit()
    c.close()

    conn = db.connect(p)
    db.insert_failed_hand(conn, "PokerStars", "raw", "why", AT)
    assert len(db.failed_hands(conn)) == 1


# --------------------------------------------------------------------------- #
# Round-4: recording a failure must be IDEMPOTENT.
#
# Found by the [R2'] mutation gate, not by these tests as first written — they
# covered clearing but never repeat-import, so a nightly cron re-importing the
# same broken file grew one row per night. Recorded here because the mutation
# checks passed while this was live: mutations prove a test is ATTACHED to the
# behavior, not that it COVERS the property space. Separate verifications.
# --------------------------------------------------------------------------- #
def test_reimporting_a_still_broken_hand_does_not_duplicate():
    """The cron case: the same bad hand, every night, forever.

    The report surface counts rows, so duplication makes one unchanged broken
    hand read as a worsening problem.
    """
    conn = db.connect()
    for at in (AT, LATER, "2026-07-21T00:00:00"):
        _session_with_one_broken_hand(conn, graded_at=at)
    assert len(failed_hand_report(conn)) == 1


def test_a_repeat_import_refreshes_rather_than_stacks():
    """The table answers "what is broken NOW", so the latest attempt wins."""
    conn = db.connect()
    raw = (FIXTURES / "ps_preflop_fold.txt").read_text()
    db.insert_failed_hand(conn, "PokerStars", raw, "old reason", AT)
    db.insert_failed_hand(conn, "PokerStars", raw, "new reason", LATER)

    rows = db.failed_hands(conn)
    assert len(rows) == 1
    assert rows[0]["reason"] == "new reason"
    assert rows[0]["imported_at"] == LATER, "the batch stamp must move too"


def test_two_different_hands_are_still_two_rows():
    """The dedup key must not collapse genuinely distinct failures."""
    conn = db.connect()
    a = (FIXTURES / "ps_preflop_fold.txt").read_text()
    b = (FIXTURES / "ps_multiway_flop.txt").read_text()
    db.insert_failed_hand(conn, "PokerStars", a, "r", AT)
    db.insert_failed_hand(conn, "PokerStars", b, "r", AT)
    assert len(db.failed_hands(conn)) == 2


def test_a_db_holding_duplicates_upgrades_cleanly(tmp_path):
    """A DB written before the constraint existed must still open.

    Unlike the duplicate-GRADINGS case, collapsing is safe here and refusing
    would be wrong: the rows are identical by key and keeping the newest IS the
    semantics being adopted, so there is nothing to lose and nothing to ask.
    """
    p = tmp_path / "dupes.db"
    conn = db.connect(p)
    conn.executescript("DROP INDEX IF EXISTS failed_hands_site_raw_uq;")
    for at in (AT, LATER):
        conn.execute(
            "INSERT INTO failed_hands(site, hand_uid, raw, reason, imported_at)"
            " VALUES ('PokerStars', NULL, 'same raw', ?, ?)", ("why", at))
    conn.commit()
    conn.close()

    conn = db.connect(p)          # must not raise
    rows = db.failed_hands(conn)
    assert len(rows) == 1
    assert rows[0]["imported_at"] == LATER, "collapse must keep the NEWEST"


def test_a_whole_file_failure_row_clears_after_a_chunked_reimport():
    """The [R2'] seam, guarded from the STORE side.

    Before 1d80de9 the importer stored the WHOLE FILE as `raw`; it now stores
    one chunk per hand. My mutation gate found that the first chunker dropped
    the file's trailing newline, so a row written under the old regime could
    never be cleared -- the summary would report a hand that imports fine,
    forever. The hunter fixed it at the source in 7c5c276 by making the chunker
    byte-faithful, which is a better remedy than the documented boundary we had
    agreed to accept.

    This pins it from the side that CARES: clearing is keyed on raw text, so a
    lossy chunker silently breaks clearing and nothing else. If the chunker ever
    stops being byte-faithful, this fails and names why.
    """
    from pokerlab.hh.pokerstars import split_pokerstars

    whole = (FIXTURES / "ps_preflop_fold.txt").read_text()
    chunk = split_pokerstars(whole)[0]
    assert chunk == whole, "a single-hand file must chunk to itself, byte for byte"

    conn = db.connect()
    db.insert_failed_hand(conn, "PokerStars", whole, "legacy", AT)
    assert db.clear_failed_hand(conn, "PokerStars", chunk) == 1
    assert db.failed_hands(conn) == []


def test_chunking_a_session_file_is_byte_faithful():
    """Rejoining the chunks must reproduce the file exactly, for the same reason."""
    from pokerlab.hh.pokerstars import split_pokerstars

    for name in ("ps_session_multi.txt", "ps_session_mixed.txt"):
        text = (FIXTURES / name).read_text()
        assert "".join(split_pokerstars(text)) == text, name


def test_a_multi_hand_pre_split_row_is_a_known_boundary():
    """[R2'] residue, stated so it is a decision rather than a discovery.

    A row written before 1d80de9 recorded "the WHOLE FILE failed". After
    splitting, that is no longer a representable outcome for a multi-hand file:
    no chunk equals the file, so no re-import can match it. Byte-faithfulness
    (7c5c276) fixes the single-hand case and cannot reach this one.

    Ruled a documented boundary rather than a migration because zero such
    databases exist. This test exists so that stays TRUE BY CHOICE — if someone
    later decides the residue is worth clearing, this fails and points at the
    decision instead of letting it look like an accident.
    """
    from pokerlab.hh.pokerstars import split_pokerstars

    whole = (FIXTURES / "ps_session_multi.txt").read_text()
    chunks = split_pokerstars(whole)
    assert len(chunks) > 1, "fixture must hold several hands"
    assert not any(c == whole for c in chunks), (
        "no chunk of a multi-hand file can equal the file")

    conn = db.connect()
    db.insert_failed_hand(conn, "PokerStars", whole, "pre-split", AT)
    for c in chunks:                      # a full successful re-import
        db.clear_failed_hand(conn, "PokerStars", c)
    assert len(db.failed_hands(conn)) == 1, "the boundary: still orphaned"


def test_a_pre_split_row_can_never_be_named_either():
    """The other half of the [R2'] boundary, and the easier half to miss.

    [R1b] recovers a hand's number from its header, and the upsert fills that
    number into an existing row — self-healing for rows recorded before the
    importer could name them. But naming matches on `(site, raw)`, the same key
    clearing uses, so it inherits the same boundary: a pre-1d80de9 row holding a
    whole multi-hand file matches no chunk and is inert to BOTH. It stays
    "unidentified" in the report permanently, not just uncleared.

    Pinned separately from the clearing case because the two live in different
    functions, so a reader can fix one and believe they have fixed both.
    """
    from pokerlab.hh.pokerstars import split_pokerstars

    whole = (FIXTURES / "ps_session_multi.txt").read_text()
    chunk = split_pokerstars(whole)[0]

    conn = db.connect()
    db.insert_failed_hand(conn, "PokerStars", whole, "legacy", AT)   # pre-split
    # a later import that DOES know the number re-records the hand it can see
    db.insert_failed_hand(conn, "PokerStars", chunk, "boom", LATER, "240000000001")

    legacy = [r for r in db.failed_hands(conn) if r["raw"] == whole]
    assert len(legacy) == 1
    assert legacy[0]["hand_uid"] is None, "the boundary: inert to naming too"


def test_a_post_split_row_IS_named_by_a_later_import():
    """The self-healing that DOES work, so the boundary above reads as a limit."""
    conn = db.connect()
    raw = (FIXTURES / "ps_preflop_fold.txt").read_text()
    db.insert_failed_hand(conn, "PokerStars", raw, "boom", AT)
    assert db.failed_hands(conn)[0]["hand_uid"] is None

    db.insert_failed_hand(conn, "PokerStars", raw, "boom", LATER, "240000000001")
    rows = db.failed_hands(conn)
    assert len(rows) == 1, "still one row"
    assert rows[0]["hand_uid"] == "240000000001", "named in place"
