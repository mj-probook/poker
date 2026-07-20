"""FastAPI drill loop (Slice E; plan §3 "UI pinned for M2": functional only).

Two JSON routes wire the browser to the drill engine — the whole UI↔engine
contract from plan §3 (`next_spot() -> Spot`, `submit_action() -> Score`):

    GET  /api/drill/next            -> Spot   (scheduler picks the category)
    POST /api/drill/answer          -> Score  (persist attempt + advance SM-2)

Answers are scored by the decision-ε rule, persisted to `drill_attempts`, and
fed to the SM-2 scheduler so the next spot resurfaces the worst leak category.
Everything is local, single-user, no auth, no build step (plan non-goals).
"""

from __future__ import annotations

import os
import random
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pokerlab.drills import generator as gen
from pokerlab.drills import scheduler as sch
from pokerlab.drills.scoring import score
from pokerlab.hh import persist
from pokerlab.hh.tiers import TIER_LABELS
from pokerlab.store import db, views
from pokerlab.types import TIER_BEST_AVAILABLE, TIER_CHART

STATIC_DIR = Path(__file__).resolve().parent / "static"

# What the operator should DO about each terminal batch cause. Three causes,
# three different actions — reporting them as one "failed" number told the user
# to retry rows that a retry cannot fix (wave-3 [P5'] follow-on):
#   * unsolvable — the spot is outside what the solver models at all.
#   * mismatched — the queue row no longer describes the hand it points at, so
#     re-draining re-derives the same wrong spot and fails identically. This is
#     the one where "retry" is actively the WRONG advice.
#   * failed     — a bug took the row down; fixing it makes a retry work.
# Keyed by `db.TERMINAL_STATUSES` — the store owns the vocabulary, this owns
# only the operator-facing copy. A test pins that the two agree, so a new
# terminal cause surfaces loudly instead of vanishing from the page.
TERMINAL_ACTIONS: dict[str, str] = {
    "unsolvable": "will not change without a solver upgrade",
    "mismatched": "re-import these hands — the queue no longer describes them",
    "failed": "a bug stopped these rows — fix it, then retry the batch",
}
# Work that will still be attempted. Stated positively rather than as
# "everything that is not terminal or done": a future in-flight status must
# count as backlog here, and subtraction would silently file it as terminal.
_IN_FLIGHT = ("pending", "running")

# The remedy for a hand the grader could not process. Its own action, because
# it is its own remedy: draining cannot help and a retry cannot help — the
# parser has to change first. Lives here for the same reason TERMINAL_ACTIONS
# does: one server-side home, carried in the payload, rendered by a page that
# never authors it.
FAILED_HANDS_ACTION = ("fix the parser, then re-import — these hands are "
                       "re-importable and clear themselves on success")
# Shown when a parse died before reaching the site's hand number.
UNIDENTIFIED_HAND = "unidentified"

# What a drill kind's ev_loss is DENOMINATED in (round-3 finding [P7']).
#
# A separate axis from `tier`/`tier_label`, deliberately. ICM drills are just
# as tier-1 chart-graded as chip-EV ones — same in-house engine, same exact
# solve — but their payoffs are ICM-$ deltas, not bb. Folding units into the
# tier vocabulary would let a future drill kind inherit one claim by asserting
# the other; they vary independently, so they are stated independently.
#
# Reporting a $-delta with a "bb" suffix was not cosmetic: the magnitudes are
# ~25x apart (1bb = 25$ on the bubble fixture), so an ICM ev_loss of 100 read
# as a catastrophic 100bb error rather than the $100 of a 1000$ pool that it
# is. Keyed by `Drill.kind`; a test pins that every kind the generator emits
# has an entry, so a new kind surfaces loudly instead of silently formatting
# as bb.
EV_UNITS: dict[str, str] = {
    "jamfold": "bb",
    "icm": "ICM-$",
}


class Answer(BaseModel):
    drill_id: str
    action: str


def _spot_json(drill: gen.Drill) -> dict:
    tc = drill.tournament
    return {
        "drill_id": drill.drill_id,
        "kind": drill.kind,
        # Which oracle graded this, in the payload itself (plan §9: tier labels
        # are load-bearing UI). Every drill the generator emits is answered by
        # the in-house chart engine, so every drill is tier 1 — stated, not
        # assumed, because a future solver-backed drill kind must not silently
        # inherit a "chart-graded" claim.
        "tier": TIER_CHART,
        "tier_label": TIER_LABELS[TIER_CHART],
        # Independent of tier — see EV_UNITS. Carried on the spot as well as on
        # the answer so the page can label the stakes before the user commits.
        "ev_unit": EV_UNITS[drill.kind],
        "position": drill.position,
        "depth_bb": drill.depth_bb,
        "hand": drill.hand_label,
        "description": drill.description,
        "legal_actions": list(drill.legal_actions),
        "pot_bb": drill.pot_bb,
        "tournament": None if tc is None else {
            "players_remaining": tc.players_remaining,
            "payouts": list(tc.payouts),
            "stacks_all": list(tc.stacks_all),
            "bb": tc.bb,
            "ante": tc.ante,
        },
    }


def _explain(result, action: str, unit: str) -> str:
    """Feedback line. `unit` names what the EV loss is denominated in.

    The unit is passed in rather than hardcoded (round-3 finding [P7']): this
    string said "bb" for every drill kind, including ICM drills whose ev_loss
    is a $-delta ~25x larger per unit. See EV_UNITS.
    """
    verdict = "Correct" if result.correct else "Incorrect"
    return (f"{verdict}. Chart plays {result.best_action} here — you chose "
            f"{action} (freq {result.chosen_frequency:.0%}, "
            f"EV loss {result.ev_loss_bb:.2f}{unit}).")


def create_app(db_path: str = ":memory:", seed: int = 0) -> FastAPI:
    app = FastAPI(title="pokerlab drills")
    population = gen.default_population()
    by_id = {d.drill_id: d for d in population}
    by_cat: dict[str, list[gen.Drill]] = {}
    for d in population:
        by_cat.setdefault(d.leak_key, []).append(d)
    default_cat = population[0].leak_key

    conn = db.connect(db_path, check_same_thread=False)
    lock = threading.Lock()
    rng = random.Random(seed)
    # Replay guard (round-2 finding [E43]): a double-click posts the same
    # drill_id twice and SM-2 counted both, inflating reps/interval off one
    # answer. Only the *immediately* repeated answer is absorbed — fetching the
    # next spot clears it, so genuinely re-practising a drill still counts.
    last: dict[str, object] = {"drill_id": None, "response": None}

    @app.get("/api/drill/next")
    def next_drill() -> dict:
        with lock:
            now = datetime.now(timezone.utc)
            cat = sch.select_next(conn, now, categories=by_cat.keys()) or default_cat
            pool = by_cat.get(cat) or population
            last["drill_id"] = None
            return _spot_json(rng.choice(pool))

    @app.post("/api/drill/answer")
    def answer(ans: Answer) -> dict:
        with lock:
            drill = by_id.get(ans.drill_id)
            if drill is None:
                raise HTTPException(404, f"unknown drill_id {ans.drill_id!r}")
            if ans.drill_id == last["drill_id"]:
                return last["response"]          # type: ignore[return-value]
            try:
                # bb_value converts the decision-ε into the drill's own payoff
                # currency ([P7']); it is 1.0 for chip-EV and the average chip
                # value for ICM. Without it ICM was graded exact-argmax.
                result = score(drill.solution, ans.action, drill.pot_bb,
                               bb_value=drill.bb_value)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            now = datetime.now(timezone.utc)
            db.insert_drill_attempt(conn, drill.leak_key, drill.kind, ans.action,
                                    result.correct, result.ev_loss_bb,
                                    now.isoformat())
            sch.schedule_attempt(conn, drill.leak_key, result.correct, now)
            payload = {
                "drill_id": drill.drill_id,
                "correct": result.correct,
                # `ev_loss` + `ev_unit`, not `ev_loss_bb`: the old key asserted
                # the unit in its NAME, which was false for ICM drills. A field
                # name is a claim like any other ([P7']).
                "ev_loss": result.ev_loss_bb,
                "ev_unit": EV_UNITS[drill.kind],
                "best_action": result.best_action,
                "chosen_frequency": result.chosen_frequency,
                "explanation": _explain(result, ans.action, EV_UNITS[drill.kind]),
            }
            last["drill_id"], last["response"] = ans.drill_id, payload
            return payload

    @app.get("/api/report")
    def report() -> dict:
        """The training loop's read surface (wave-3 [P5']).

        Deliberately shaped so the two grading regimes CANNOT be rendered as one
        list: `leaks` is the exact-tier EV-loss ranking, `tier3` is its own
        object carrying its own honesty label. Plan §9 — EV-loss stats never mix
        tiers, and the label travels with the data rather than living in the
        template, so no future page can drop it.
        """
        with lock:
            counts = conn.execute(
                "SELECT (SELECT COUNT(*) FROM imported_hands)            AS hands,"
                "       (SELECT COUNT(*) FROM gradings)                  AS graded,"
                "       (SELECT COUNT(*) FROM batch_queue"
                "         WHERE status IN (?, ?))                        AS queued",
                _IN_FLIGHT,
            ).fetchone()
            session = dict(counts)
            # PARTIAL means "work is still coming" — and now it can mean it.
            # `queued` counts only in-flight rows, so a terminal row no longer
            # pins the session partial forever while the banner promises numbers
            # that will never move (wave-3 [P5'] follow-on).
            session["partial"] = session["queued"] > 0
            # Terminal rows are not backlog, but they are not nothing either:
            # each is reported with the action that actually clears it.
            marks = ",".join("?" * len(db.TERMINAL_STATUSES))
            blocked = conn.execute(
                "SELECT status, COUNT(*) AS count FROM batch_queue"
                f" WHERE status IN ({marks})"
                " GROUP BY status ORDER BY count DESC, status",
                db.TERMINAL_STATUSES,
            )
            session["blocked"] = [
                dict(r) | {"action": TERMINAL_ACTIONS[r["status"]]}
                for r in blocked
            ]
            # PLAN §8 M4 exit: failed hands are "isolated and surfaced in the
            # session summary — never silently dropped". A count alone would
            # recreate the not-actionable defect the terminal-cause split just
            # fixed ("3 failed" without WHICH), so every failure carries its
            # identity and its reason, and the section carries its remedy.
            #
            # Deliberately NOT folded into `partial`: draining will never change
            # a failed hand, so `partial` would go back to being permanently
            # true under a banner promising numbers that cannot move. Also kept
            # apart from `skipped` — "already imported, nothing to do" and "you
            # lost a hand" are different events, and collapsing remedies is the
            # defect caught in retry_failed_batch.
            failures = persist.failed_hand_report(conn)
            session["failed"] = len(failures)
            session["failed_action"] = FAILED_HANDS_ACTION
            session["failed_hands"] = [
                {"site": f["site"], "hand_uid": f["hand_uid"],
                 "reason": f["reason"], "imported_at": f["imported_at"],
                 # A parse can die before it reaches the hand number, so the uid
                 # is nullable. Unknown is not absent: it renders as a stated
                 # "unidentified", never a blank and never a fabricated id —
                 # the same rule as a NULL population_frequency showing "no
                 # baseline" rather than 0%.
                 "label": f["hand_uid"] or UNIDENTIFIED_HAND}
                for f in failures
            ]
            return {
                "session": session,
                "leaks": views.hh_leak_report(conn, limit=5),
                "tier3": {
                    "label": TIER_LABELS[TIER_BEST_AVAILABLE],
                    "rows": views.tier3_frequency_report(conn),
                },
                "gates": {
                    "ev_loss_trend":
                        views.ev_loss_per_100_by_category_over_time(conn),
                    "accuracy_by_kind": views.accuracy_by_kind(conn),
                },
            }

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/report")
    def report_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "report.html")

    app.state.conn = conn
    app.state.population = population
    return app


def main() -> None:  # pragma: no cover - convenience runner (pokerlab-web)
    import uvicorn

    app = create_app(db_path=os.environ.get("POKERLAB_DB", "pokerlab.db"))
    uvicorn.run(app, host=os.environ.get("POKERLAB_HOST", "127.0.0.1"),
                port=int(os.environ.get("POKERLAB_PORT", "8000")))
