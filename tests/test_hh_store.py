"""Slice F step 3: persist gradings + batch_queue and query leak reports.

Reuses Slice E's `store` (schema, insert_grading, tier-3 honesty trigger) and
`store.views`; nothing here duplicates that module. A stub solver stands in for
the (not-yet-landed) real solver behind the injected `Solver` seam.
"""

from pathlib import Path

from pokerlab.hh.persist import drain_batch_queue, persist_session
from pokerlab.hh.ggpoker import parse_ggpoker
from pokerlab.hh.pokerstars import parse_pokerstars
from pokerlab.hh.population import load_population
from pokerlab.hh.report import grade_session
from pokerlab.store import db
from pokerlab.store.views import hh_leak_report, tier3_frequency_report
from pokerlab.types import Solution

FIXTURES = Path(__file__).parent / "fixtures" / "hh"
AT = "2026-07-19T00:00:00"

_PS = ["ps_preflop_fold.txt", "ps_multiway_flop.txt",
       "ps_allin_sidepot.txt", "ps_ante_hu.txt"]
_GG = ["gg_preflop_3bet.txt", "gg_allin_jam.txt", "gg_sb_fold_leak.txt"]

_STUB_SOLUTION = Solution(
    actions={"check": (0.0, 0.5), "bet": (0.1, 0.3), "call": (0.05, 0.15),
             "raise": (0.02, 0.0), "fold": (-0.5, 0.05)},
    range_ctx="stub:hu-postflop", source="solver")


def _load():
    raws, parsed = [], []
    for f in _PS:
        t = (FIXTURES / f).read_text(); raws.append(t)
        parsed.append(parse_pokerstars(t))
    for f in _GG:
        t = (FIXTURES / f).read_text(); raws.append(t)
        parsed.append(parse_ggpoker(t))
    return parsed, raws


def _persisted_session():
    parsed, raws = _load()
    report = grade_session(parsed, population=load_population())
    conn = db.connect()
    counts = persist_session(conn, parsed, report, graded_at=AT, raw_texts=raws)
    return conn, report, counts


def test_persist_writes_hands_gradings_and_queues_tier2() -> None:
    conn, report, counts = _persisted_session()
    assert counts["imported"] == len(_PS) + len(_GG)
    assert counts["graded"] == len(report.exact) + len(report.approx)
    # every tier-2 miss became a pending batch row (report is partial)
    assert counts["queued"] == len(report.tier2_pending)
    assert len(db.pending_batch(conn)) == counts["queued"] > 0


def test_tier3_rows_persist_with_null_ev_loss() -> None:
    conn, _report, _counts = _persisted_session()
    tier3 = [g for g in db.gradings(conn) if g["tier"] == 3]
    assert tier3, "expected multiway tier-3 gradings"
    assert all(g["ev_loss"] is None for g in tier3)  # honesty trigger satisfied


def test_batch_worker_drains_queue_and_grades_tier2() -> None:
    conn, _report, counts = _persisted_session()

    def solver(spot_key: str, decision) -> Solution:
        return _STUB_SOLUTION  # always a hit

    result = drain_batch_queue(conn, solver, graded_at=AT)
    assert result["done"] == counts["queued"] and result["failed"] == 0
    assert db.pending_batch(conn) == []
    assert all(r["status"] == "done" for r in db.batch_rows(conn))
    # tier-2 gradings now exist and carry an ev_loss
    tier2 = [g for g in db.gradings(conn) if g["tier"] == 2]
    assert tier2 and all(g["ev_loss"] is not None for g in tier2)


def test_solver_miss_marks_row_failed() -> None:
    conn, _report, counts = _persisted_session()
    result = drain_batch_queue(conn, lambda spot, d: None, graded_at=AT)
    assert result["failed"] == counts["queued"] and result["done"] == 0
    assert all(r["status"] == "failed" for r in db.batch_rows(conn))


def test_leak_report_ranks_tiers_1_2_and_lists_tier3_separately() -> None:
    conn, _report, _counts = _persisted_session()
    drain_batch_queue(conn, lambda spot, d: _STUB_SOLUTION, graded_at=AT)

    leaks = hh_leak_report(conn, limit=5)
    assert leaks, "expected ranked leaks"
    # the SB open-fold of A7o at 10bb is the biggest leak in the session; its
    # canonical category is the SB-jam 10bb SPOT (the hero's fold does not
    # change the category — that is what lets it join the drill of the same key)
    assert leaks[0]["leak_key"] == "SBjam|preflop|jam|10"
    assert leaks[0]["ev_loss_per_100"] > 100  # ~114 bb/100
    # ranking is monotone non-increasing in the stated metric
    metric = [row["ev_loss_per_100"] for row in leaks]
    assert metric == sorted(metric, reverse=True)

    # tier-3 is reported separately and never appears in the EV-loss ranking
    t3 = tier3_frequency_report(conn)
    t3_keys = {r["leak_key"] for r in t3}
    assert t3_keys and not (t3_keys & {row["leak_key"] for row in leaks})


# --------------------------------------------------------------------------- #
# Round-1 finding [10]: batch-drain isolation + startup recovery. A solver that
# raises must fail its own row only, and rows stranded at 'running' by a crashed
# earlier drain must be recovered rather than stuck forever.
# --------------------------------------------------------------------------- #
def test_raising_solver_fails_only_its_own_row() -> None:
    conn, _report, counts = _persisted_session()
    assert counts["queued"] >= 2, "need >1 queued row to prove isolation"
    # the drain walks the queue in id order, so the first row is the casualty
    first_id = db.pending_batch(conn)[0]["id"]
    calls = []

    def solver(spot_key: str, decision):
        calls.append(spot_key)
        if len(calls) == 1:
            raise RuntimeError("solver exploded")
        return _STUB_SOLUTION

    result = drain_batch_queue(conn, solver, graded_at=AT)

    # the drain kept going after the explosion
    assert len(calls) == counts["queued"]

    assert result["failed"] == 1
    assert result["done"] == counts["queued"] - 1
    statuses = {r["id"]: r["status"] for r in db.batch_rows(conn)}
    assert statuses[first_id] == "failed"
    assert all(s == "done" for rid, s in statuses.items() if rid != first_id)


def test_stranded_running_rows_are_recovered_on_drain() -> None:
    conn, _report, counts = _persisted_session()
    # simulate a drain that died mid-row
    stranded = db.pending_batch(conn)[0]["id"]
    db.set_batch_status(conn, stranded, "running")
    assert len(db.pending_batch(conn)) == counts["queued"] - 1

    result = drain_batch_queue(conn, lambda spot, d: _STUB_SOLUTION, graded_at=AT)

    assert result["recovered"] == 1
    # the stranded row was picked back up and completed with the rest
    assert result["done"] == counts["queued"]
    assert all(r["status"] == "done" for r in db.batch_rows(conn))
