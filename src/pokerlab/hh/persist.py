"""Store persistence for the HH pipeline (Slice F step 3; impl doc §3 / §2).

Bridges the pure `hh.report` grading to Slice E's `store`:
  * `persist_session` writes each hand to `imported_hands`, every graded
    decision to `gradings` (tier-3 rows carry NULL ev_loss — the honesty
    trigger stays satisfied), and enqueues every tier-2 *miss* into
    `batch_queue` (report marked partial by the pending count).
  * `drain_batch_queue` is the batch worker: for each pending row it calls the
    (injected) solver, and on a hit re-derives the decision from the stored raw
    hand and grades it tier-2, marking the row done/failed.

The solver is injected (stub in tests, real solver in integration), so nothing
here depends on Slice D landing.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from pokerlab.hh import ggpoker, pokerstars
from pokerlab.hh.decisions import Decision, extract_decisions
from pokerlab.hh.grade import grade_tier2
from pokerlab.hh.model import ParsedHand
from pokerlab.hh.report import SessionReport
from pokerlab.store import db
from pokerlab.types import Solution, TIER_SOLVER

# The drain solver receives the queued spot_key AND the re-derived decision, so
# a real solver can build the correct subgame (board/pot/stack) and pick the
# hero's hand class; a stub can ignore both. Returns None on a solve miss.
Solver = Callable[[str, Decision], Solution | None]

_PARSERS = {
    "PokerStars": pokerstars.parse_pokerstars,
    "GGPoker": ggpoker.parse_ggpoker,
}


def spot_key(d: Decision) -> str:
    """Solve-spot identifier for the tier-2 library / batch queue (action-free)."""
    return f"{d.formation}|{d.street}"


def parse_by_site(site: str, raw: str) -> ParsedHand:
    try:
        return _PARSERS[site](raw)
    except KeyError:
        raise ValueError(f"no parser for site {site!r}") from None


def _summary_json(ph: ParsedHand) -> str:
    return json.dumps({
        "site": ph.site,
        "hand_id": ph.hand_id,
        "hero": ph.hero,
        "seats": len(ph.setup.stacks),
        "stacks": list(ph.setup.stacks),
    })


def persist_session(conn, parsed_hands: list[ParsedHand], report: SessionReport,
                    *, graded_at: str, raw_texts: list[str] | None = None) -> dict:
    """Persist a graded session atomically. Counts {imported, graded, queued, skipped}.

    One transaction: either the whole session lands or none of it does, so a
    failure part-way through cannot leave a half-written session behind. Hands
    already imported (same site + HH hand number) are skipped along with their
    gradings, so re-importing a file never double-counts into the leak stats
    (round-1 finding [11]).
    """
    n_graded = n_queued = n_skipped = 0
    try:
        hand_ids: list[int | None] = []
        for i, ph in enumerate(parsed_hands):
            raw = raw_texts[i] if raw_texts is not None else ""
            hid = db.insert_imported_hand(
                conn, ph.site, raw, _summary_json(ph), graded_at,
                ph.hand_id, commit=False)
            if hid is None:
                n_skipped += 1
            hand_ids.append(hid)

        for gd in report.graded:
            hid = hand_ids[gd.hand_index]
            if hid is None:
                continue  # duplicate hand: its gradings are already stored
            g = gd.grading
            if g.graded:
                db.insert_grading(conn, hid, g.decision_index, g.tier, g.chosen,
                                  g.best or "", g.ev_loss, g.leak_key, graded_at,
                                  commit=False)
                n_graded += 1
            elif g.tier == TIER_SOLVER:
                db.enqueue_batch(conn, spot_key(gd.decision), hid,
                                 g.decision_index, commit=False)
                n_queued += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {"imported": sum(h is not None for h in hand_ids),
            "graded": n_graded, "queued": n_queued, "skipped": n_skipped}


def drain_batch_queue(conn, solver: Solver, *, graded_at: str) -> dict:
    """Batch worker: solve each pending spot, grade it tier-2, mark row done.

    A solver miss (returns None) marks the row failed and leaves it ungraded.
    So does a row that *raises* — a re-parse failure, a decision index that no
    longer resolves, or a solver blowing up must cost that row only, never the
    rest of the backlog (round-1 finding [10]).

    Rows stranded at 'running' by an earlier crashed drain are recovered to
    'pending' first, so the backlog cannot silently leak work.

    Returns counts {done, failed, recovered}.
    """
    recovered = db.recover_running_batch(conn)
    done = failed = 0
    for row in db.pending_batch(conn):
        db.set_batch_status(conn, row["id"], "running")
        try:
            hand = db.get_imported_hand(conn, row["hand_id"])
            parsed = parse_by_site(hand["site"], hand["raw"])
            d = extract_decisions(parsed)[row["decision_idx"]]
            solution = solver(row["spot_key"], d)
        except Exception:  # noqa: BLE001 - per-row isolation boundary
            db.set_batch_status(conn, row["id"], "failed")
            failed += 1
            continue
        if solution is None:
            db.set_batch_status(conn, row["id"], "failed")
            failed += 1
            continue
        g = grade_tier2(d, solution)
        db.insert_grading(conn, row["hand_id"], d.index, g.tier, g.chosen,
                          g.best or "", g.ev_loss, g.leak_key, graded_at)
        db.set_batch_status(conn, row["id"], "done")
        done += 1
    return {"done": done, "failed": failed, "recovered": recovered}
