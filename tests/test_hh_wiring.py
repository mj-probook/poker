"""Round-3 findings [P1'] / [P3']: the M4->M2 wiring, end to end.

Separate from test_hh_store.py (which owns the persist/drain seam itself)
because these assert the WIRING between modules — that importing a session is
what makes its leaks drillable, and that tier 3's frequency signal survives the
trip to the store and back out through the report.

Shares test_hh_store's fixture session loader rather than re-deriving it: there
is one canonical graded session and both files must mean the same thing by it.
"""

from datetime import datetime, timedelta

import pytest

from pokerlab.drills.scheduler import select_next
from pokerlab.store import db
from pokerlab.store.views import hh_leak_report, tier3_frequency_report

from tests.test_hh_store import AT, _persisted_session


# --------------------------------------------------------------------------- #
# Round-3 finding [P1'] (P0): the M4->M2 leak->drill join existed only in test
# code — `persist_session` wrote no `sr_state`, so the scheduler could never
# learn about a leak the HH import had just detected. The join has to happen in
# PRODUCTION code: importing a session is what makes its leaks drillable.
# --------------------------------------------------------------------------- #
def test_persist_seeds_sr_state_from_the_sessions_exact_tier_leaks() -> None:
    conn, report, _counts = _persisted_session()

    seeded = {r["leak_key"] for r in db.all_sr_state(conn)}
    exact = {gd.grading.leak_key for gd in report.exact}
    assert exact, "fixture session must contain exact-tier gradings"
    assert seeded == exact, "every exact-tier leak becomes a drillable category"


def test_importing_a_session_makes_its_worst_leak_the_next_drill() -> None:
    """The P1' invariant: the served category IS the top exact-tier leak.

    Asserted as an invariant over whatever the fixtures grade to, never as the
    pre-fix symptom (a hard-coded winning key) — /tmp/pl README's warning that a
    check encoding the old world becomes a regression detector for the fix.
    """
    conn, _report, counts = _persisted_session()
    assert counts["seeded"] > 0

    # A DECOY that makes this test able to fail. The seeded categories are all
    # due at the same instant with zero drill attempts, so error-rate and
    # due-date — the only keys the pre-fix scheduler had — cannot separate them,
    # and the right answer would win by alphabetical/insertion accident. This
    # category is trivial in EV but BOTH sorts first and has been due longest, so
    # every pre-fix tiebreaker points at it: only a ranking that actually reads
    # HH EV-loss picks the real leak.
    hid = db.insert_imported_hand(conn, "PokerStars", "raw", "{}", AT, "decoy")
    db.insert_grading(conn, hid, 0, 1, "jam", "jam", 0.01,
                      "AAdecoy|preflop|jam|10", AT)
    db.seed_sr_state(conn, "AAdecoy|preflop|jam|10", 2.5, 0.0, 0,
                     "2026-07-18T00:00:00")

    leaks = hh_leak_report(conn, limit=5)
    now = datetime.fromisoformat(AT) + timedelta(minutes=1)
    served = select_next(conn, now, categories=[r["leak_key"] for r in leaks])
    assert served == leaks[0]["leak_key"] != "AAdecoy|preflop|jam|10"


# --------------------------------------------------------------------------- #
# Round-3 finding [P3'] (P1): tier 3's ONLY honest output is the frequency
# deviation, and persist discarded it — the grader computed a population
# frequency and a flag, wrote neither, and the report could show nothing but a
# bare count. Plan §5.3 promises the flag as the tier-3 signal.
# --------------------------------------------------------------------------- #
def test_tier3_rows_persist_their_frequency_and_flags() -> None:
    conn, report, _counts = _persisted_session()
    approx = [gd for gd in report.approx if gd.grading.tier == 3]
    assert approx, "fixture session must contain multiway tier-3 gradings"

    stored = {(g["leak_key"], g["chosen"]): g
              for g in db.gradings(conn) if g["tier"] == 3}
    for gd in approx:
        g = stored[(gd.grading.leak_key, gd.grading.chosen)]
        assert g["frequency"] == pytest.approx(gd.grading.frequency)
        assert g["ev_loss"] is None          # honesty rule still holds
        assert g["flags"] == ",".join(gd.grading.flags)


def test_tier3_report_surfaces_frequencies_and_flags_not_bare_counts() -> None:
    conn, _report, _counts = _persisted_session()
    rows = tier3_frequency_report(conn)
    assert rows, "expected tier-3 rows"
    for r in rows:
        assert set(r) >= {"leak_key", "chosen", "decisions",
                          "population_frequency", "flags"}
        assert "ev_loss" not in r            # never, at this tier
        assert isinstance(r["flags"], list)

    # The fixture session's only multiway spot has no population baseline
    # (`6max:LJ|flop` is absent from population/default.toml), so it exercises
    # the None branch — "unknown", never to be read as a 0% frequency. The
    # deviation flag itself is proven on a spot the table DOES cover.
    assert rows[0]["population_frequency"] is None and rows[0]["flags"] == []

    hid = db.insert_imported_hand(conn, "PokerStars", "raw", "{}", AT, "rare")
    db.insert_grading(conn, hid, 0, 3, "raise", "", None,
                      "6max:BB|flop|raise|-", AT,
                      frequency=0.03, flags=["freq_deviation"])
    flagged = next(r for r in tier3_frequency_report(conn)
                   if r["leak_key"] == "6max:BB|flop|raise|-")
    assert flagged["population_frequency"] == pytest.approx(0.03)
    assert flagged["flags"] == ["freq_deviation"]
