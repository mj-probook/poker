"""Store persistence for the HH pipeline (Slice F step 3; impl doc §3 / §2).

Bridges the pure `hh.report` grading to Slice E's `store`:
  * `persist_session` writes each hand to `imported_hands`, every graded
    decision to `gradings` (tier-3 rows carry NULL ev_loss — the honesty
    trigger stays satisfied), and enqueues every tier-2 *miss* into
    `batch_queue` (report marked partial by the pending count).
  * `drain_batch_queue` is the batch worker: for each pending row it re-derives
    the decision from the stored raw hand, passes it to the injected solver,
    grades tier-2 on a hit, and marks the row done/failed. Re-deriving comes
    first — the solver is called *with* a decision, not consulted before one
    exists.

Three invariants the seam depends on, all load-bearing:
  * **Atomic.** A tier-2 grading and the row-status transition that describes it
    commit together or not at all, so a crash can never leave a row marked
    'done' with no grading behind it (or the reverse).
  * **Deduped.** `imported_hands` carries UNIQUE(site, hand_uid); a re-imported
    session is skipped rather than double-counted, and a hand that failed to
    replay claims no uid at all — so it re-imports cleanly once the parser is
    fixed.
  * **Recoverable.** Rows stranded at 'running' by a crashed drain are returned
    to 'pending' at the start of the next drain. 'failed' means permanently
    unsolvable (the solver raised `UnsolvableSpot`), never "transiently missed" —
    a miss stays pending backlog and remains drainable.

The solver is injected (stub in tests, real solver in integration), so nothing
here depends on Slice D landing.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from pokerlab.drills.scheduler import DEFAULT_EASINESS
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


class UnsolvableSpot(Exception):
    """Raised by a drain solver to mark a spot permanently unsolvable.

    The contract that lets the drain tell "no solution yet" (return None -> the
    row stays queued) apart from "no solution is possible for this spot as
    posed" (raise this -> the row is closed, visibly). Without the distinction
    the two collapse into one terminal status and real backlog is lost
    (wave-2 [E15]).
    """


_PARSERS = {
    "PokerStars": pokerstars.parse_pokerstars,
    "GGPoker": ggpoker.parse_ggpoker,
}


def spot_key(d: Decision) -> str:
    """Coarse queue label for the tier-2 backlog: `formation|street`.

    Deliberately NOT a `types.SpotKey` (which also carries stack and board
    buckets): the drain hands the solver this label *and* the re-derived
    decision, and the decision is what carries board/pot/stack. The label only
    has to group the backlog, so it stays cheap to compute pre-solve.
    """
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
    """Persist a graded session atomically.

    Counts {imported, graded, queued, skipped, failed, seeded}.

    Importing is also what wires M4 to M2: every exact-tier (1/2) leak the
    session produced is seeded into `sr_state` due immediately, so the drill
    scheduler can serve it. That join used to live only in test code, which made
    the whole "your hands pick your drills" promise vacuous (wave-3 [P1']).

    One transaction: either the whole session lands or none of it does, so a
    failure part-way through cannot leave a half-written session behind. Hands
    already imported (same site + HH hand number) are skipped along with their
    gradings, so re-importing a file never double-counts into the leak stats
    (round-1 finding [11]).

    Hands the grader could not process (`report.failed_hands`) are NOT written
    at all — importantly they do not claim their `(site, hand_uid)`, so once the
    parser bug that broke them is fixed, re-importing the file picks them up.
    Writing them would have made the dedup permanent (wave-2 [E16]). A hand
    that failed only *some* of its decisions is dropped whole for the same
    reason: it stays re-importable, rather than contributing partial stats that
    can never be completed.
    """
    n_graded = n_queued = n_skipped = 0
    seeded: set[str] = set()
    failed_idx = {fh.hand_index for fh in report.failed_hands}
    n_failed = len(failed_idx)
    try:
        hand_ids: list[int | None] = []
        for i, ph in enumerate(parsed_hands):
            if i in failed_idx:
                hand_ids.append(None)
                continue
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
                continue  # duplicate (already stored) or failed (not stored)
            g = gd.grading
            if g.graded:
                db.insert_grading(conn, hid, g.decision_index, g.tier, g.chosen,
                                  # '' is the no-single-best sentinel: tier 3
                                  # never names a best action (schema: best TEXT
                                  # NOT NULL), and neither does an ungraded spot
                                  g.best or "", g.ev_loss, g.leak_key, graded_at,
                                  commit=False, frequency=g.frequency,
                                  flags=g.flags)
                n_graded += 1
                if g.ev_loss is not None:
                    # THE LEAK->DRILL JOIN (wave-3 [P1']). An exact-tier
                    # grading is evidence of a real, EV-priced leak, so the
                    # import itself makes that category drillable — due at the
                    # session's own timestamp, i.e. immediately. Tier 3 is
                    # excluded on purpose: it has no ev_loss, so it can neither
                    # be ranked against the exact tiers nor honestly claimed as
                    # a leak (plan §5.3).
                    seeded.add(g.leak_key)
            elif g.tier == TIER_SOLVER:
                db.enqueue_batch(conn, spot_key(gd.decision), hid,
                                 g.decision_index, commit=False)
                n_queued += 1

        for leak_key in sorted(seeded):
            # Due at `graded_at` — the session's own clock, so seeding is as
            # deterministic as the rest of persist and the leak is drillable the
            # moment the import finishes. A category already in `sr_state` keeps
            # its learned easiness/interval and is only re-opened (see
            # db.seed_sr_state); the ORDER the scheduler then serves these in is
            # a query, not a stored rank (views.category_priority).
            db.seed_sr_state(conn, leak_key, DEFAULT_EASINESS, 0.0, 0,
                             graded_at, commit=False)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {"imported": sum(h is not None for h in hand_ids),
            "graded": n_graded, "queued": n_queued, "skipped": n_skipped,
            "failed": n_failed, "seeded": len(seeded)}


def drain_batch_queue(conn, solver: Solver, *, graded_at: str) -> dict:
    """Batch worker: solve each pending spot, grade it tier-2, mark row done.

    Row outcomes, and why they differ (wave-2 [E15]):

      * **hit** -> graded, 'done'.
      * **miss** (solver returns None) -> left 'pending'. A miss is the normal
        state of a backlog — the library just does not have that spot yet — so
        it must stay drainable. Marking misses 'failed' silently dropped every
        spot queued before the solver could handle it.
      * **unsolvable** (solver raises `UnsolvableSpot`) -> 'failed'. Terminal on
        purpose: retrying cannot help. Counted separately so it is visible.
      * **error** (anything else raises) -> 'failed', and only that row; a bug
        must never cost the rest of the backlog (round-1 finding [10]).

    'failed' is not a black hole: `db.retry_failed_batch` reopens those rows
    deliberately once the cause is fixed.

    Rows stranded at 'running' by an earlier crashed drain are recovered to
    'pending' first, so the backlog cannot silently leak work.

      * **mismatched** (the re-derived decision is not the one queued) ->
        'failed'. Terminal and counted apart from a solver failure, because it
        means the queue and the hand have diverged, not that the solve is hard
        (wave-3 [B2]).

    Returns counts {done, missed, unsolvable, failed, mismatched, recovered}.
    """
    recovered = db.recover_running_batch(conn)
    done = missed = unsolvable = failed = mismatched = 0
    for row in db.pending_batch(conn):
        db.set_batch_status(conn, row["id"], "running")
        try:
            hand = db.get_imported_hand(conn, row["hand_id"])
            parsed = parse_by_site(hand["site"], hand["raw"])
            d = extract_decisions(parsed)[row["decision_idx"]]
        except Exception:  # noqa: BLE001 - per-row isolation boundary
            # No UnsolvableSpot arm here: that exception is the *solver's*
            # contract, so re-deriving cannot raise it. A handler for it would
            # document a check this stage never performs.
            db.set_batch_status(conn, row["id"], "failed")
            failed += 1
            continue

        # The row is re-derived by INDEX against a hand parsed now, not the
        # decision that was queued. If the parser shifted shape in between —
        # adding or reordering a decision, which is exactly what the [E6]/[E33]
        # parser fixes did — index N is quietly a different spot, and a preflop
        # decision gets solved and stored as tier-2 (wave-3 [B2]). The row
        # already carries `formation|street`, so identity is free to check.
        # Mismatch is terminal and loud: the backlog no longer describes the
        # hand, so grading it would write a confidently wrong row.
        if spot_key(d) != row["spot_key"]:
            db.set_batch_status(conn, row["id"], "mismatched")
            mismatched += 1
            continue

        try:
            solution = solver(row["spot_key"], d)
        except UnsolvableSpot:
            db.set_batch_status(conn, row["id"], "unsolvable")
            unsolvable += 1
            continue
        except Exception:  # noqa: BLE001 - per-row isolation boundary
            db.set_batch_status(conn, row["id"], "failed")
            failed += 1
            continue
        if solution is None:
            db.set_batch_status(conn, row["id"], "pending")  # still drainable
            missed += 1
            continue
        # GRADING IS INSIDE THE BOUNDARY. [10] said a bug must never cost the
        # rest of the backlog, but the boundary only wrapped the solve half:
        # anything `grade_tier2` raised escaped the loop, skipping every
        # remaining row and stranding this one at 'running'. It raises for real
        # — a solution whose action space does not contain the hero's actual
        # action is a ValueError out of `drills.scoring.score`.
        #
        # The grading and the row's 'done' mark stay ONE transaction: a crash
        # between them would otherwise leave the row 'running' with its grading
        # already stored, and recovery would reopen it for a second, duplicate
        # grading. Committing them together means that state cannot exist, so
        # recovery stays safe as written (wave-2 [E14]).
        try:
            g = grade_tier2(d, solution)
            db.insert_grading(conn, row["hand_id"], d.index, g.tier, g.chosen,
                              # '' = no single best action (see persist_session)
                              g.best or "", g.ev_loss, g.leak_key, graded_at,
                              commit=False,
                              # the score already computed how often the solver
                              # plays the hero's action; dropping it would throw
                              # away the honest half of a tier-2 grade
                              frequency=g.frequency)
            db.set_batch_status(conn, row["id"], "done", commit=False)
            conn.commit()
        except UnsolvableSpot:
            conn.rollback()
            db.set_batch_status(conn, row["id"], "unsolvable")
            unsolvable += 1
            continue
        except Exception:  # noqa: BLE001 - per-row isolation boundary
            # Roll the partial write back FIRST so a failed row never leaves a
            # grading behind, then close the row and keep draining: the whole
            # point of [10] is that one bad row costs one row.
            conn.rollback()
            db.set_batch_status(conn, row["id"], "failed")
            failed += 1
            continue
        done += 1
    return {"done": done, "missed": missed, "unsolvable": unsolvable,
            "failed": failed, "mismatched": mismatched, "recovered": recovered}
