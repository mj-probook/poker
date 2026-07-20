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
from pokerlab.hh.tiers import TIER_LABELS
from pokerlab.store import db, views
from pokerlab.types import TIER_BEST_AVAILABLE, TIER_CHART

STATIC_DIR = Path(__file__).resolve().parent / "static"


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


def _explain(result, action: str) -> str:
    verdict = "Correct" if result.correct else "Incorrect"
    return (f"{verdict}. Chart plays {result.best_action} here — you chose "
            f"{action} (freq {result.chosen_frequency:.0%}, "
            f"EV loss {result.ev_loss_bb:.2f}bb).")


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
                result = score(drill.solution, ans.action, drill.pot_bb)
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
                "ev_loss_bb": result.ev_loss_bb,
                "best_action": result.best_action,
                "chosen_frequency": result.chosen_frequency,
                "explanation": _explain(result, ans.action),
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
                "         WHERE status IN ('pending', 'running'))        AS queued,"
                "       (SELECT COUNT(*) FROM batch_queue"
                "         WHERE status = 'failed')                       AS failed"
            ).fetchone()
            session = dict(counts)
            # A queued row means some decision in this session has no answer key
            # yet, which is exactly what plan §5.3 calls a PARTIAL session.
            session["partial"] = session["queued"] > 0
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
