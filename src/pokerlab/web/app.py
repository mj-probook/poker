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
from pokerlab.drills.scoring import MIX_FREQ, epsilon, score
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

# Names the DB-wide backlog's scope in the SERVER's words, next to the number
# it describes. Deliberately says nothing about "this session": the round-4 [9]
# defect was one word ("partial") answering for two different scopes, and a
# label the page could author itself would let that drift straight back in.
IN_FLIGHT_LABEL = "queued across all imports — not only this session"

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
    "ring": "bb",
}

# WHICH ORACLE decided the best action, in the feedback line's own words
# (round-4 finding [10]). `_explain` said "Chart plays X" for every drill —
# true today only because every drill is chart-sourced, which is precisely the
# property the tier-source pin exists to guarantee and precisely what stops
# holding the moment a solver-backed kind ships. That drill would have told the
# user "Chart plays X" about a CFR solve: a false provenance claim on the one
# line they read after every answer.
#
# A third axis alongside `tier` and `ev_unit`, and independent of both — the
# tier says how exact the answer is, the unit says what the loss is measured
# in, this says who produced it. Keyed by `Solution.source` (types.py owns that
# vocabulary); a test pins that every source the generator emits has an entry.
#
# Only sources that actually exist are listed. Pre-writing copy for oracles
# nobody has built would be the same unwarranted-blanket-claim defect one level
# up — a new source must fail loudly here, not inherit a sentence written for a
# different one.
SOURCE_VERBS: dict[str, str] = {
    "chart": "Chart plays",
}

# What a tier-2 approximation actually means, in the server's words (PLAN §5.3
# rev-3.2; round-4 [1]). The wording is load-bearing in a direction that is easy
# to get backwards: the SOLVE is exact — the accuracy bar applies to it in full
# — and the INPUTS are the approximation. Copy that said only "approximate"
# would invite the reader to discount the number as sloppy, when what is
# approximate is the game the solver was handed, not the quality of its answer.
#
# Shown per leak beside a COUNT of affected decisions, never as a badge on the
# section: a leak_key mixes tiers, so some of its decisions carry an exact chart
# reference and some do not. A blanket label would assert for every row a
# property only some rows have — the defect f0c1bd8 exists to prevent, and the
# reason the store half reports a count rather than a boolean.
APPROX_CAVEAT = (
    "These solves are exact for the game they were handed — both players' "
    "ranges assumed uniform and stacks symmetric — not for the hand as "
    "actually played. Range modeling from the real line is future work."
)


class Answer(BaseModel):
    drill_id: str
    action: str


def _seats_json(drill: gen.Drill) -> list[dict] | None:
    """Every seat's state, in action order — a formation FACT, not layout.

    Only for drills whose seat attribution is a rules fact (`drill.table`
    non-empty: HU + 9-max ring). Everyone before the hero folded (that is
    what first-in / facing-a-jam means), except the seat that jammed; seats
    after the hero have not acted. ICM multiway returns None — its payload
    does not attribute stacks to seats, and a seats array would fabricate
    exactly that attribution.
    """
    if not drill.table:
        return None
    hero_i = drill.table.index(drill.position)
    jam_i = drill.table.index(drill.versus) if drill.versus else hero_i
    out = []
    for i, pos in enumerate(drill.table):
        if i == hero_i:
            state = "hero"
        elif drill.versus and i == jam_i:
            state = "all-in"
        elif i < hero_i:
            state = "folded"
        else:
            state = "live"
        out.append({"pos": pos, "state": state})
    return out


def _action_line(drill: gen.Drill) -> str:
    """The prior action in the server's words. Never enumerates the hero's
    options (the button row contains distractors)."""
    if drill.versus:
        return f"{drill.versus} is all-in for {drill.depth_bb:g}bb — action on you"
    if drill.table and drill.position != drill.table[0]:
        return "Folded to you — action on you"
    return "Action on you"


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
        # The PRIOR ACTION as on-table facts (user report: the jam lived only
        # in the prompt sentence under the felt, so the scene showed a
        # decision with no action). Server-authored — the page draws these
        # words and never derives its own account of what happened.
        "action_line": _action_line(drill),
        "facing_allin": bool(drill.versus),
        # Per-seat states in action order where attribution is a rules fact,
        # None where it is not (ICM) — see _seats_json.
        "seats": _seats_json(drill),
        "legal_actions": list(drill.legal_actions),
        # Submittable DISTRACTORS outside the solved game (generator.Drill):
        # rendered as buttons, graded as framework deviations with no EV
        # number. Kept apart from legal_actions so the page can never claim
        # the chart priced them.
        "off_tree_actions": list(drill.off_tree_actions),
        # Server-authored sentence for spots whose action set is complete at
        # two (BB facing an all-in) — the page renders it, never authors it.
        "action_note": drill.action_note,
        "pot_bb": drill.pot_bb,
        "tournament": None if tc is None else {
            "players_remaining": tc.players_remaining,
            "payouts": list(tc.payouts),
            "stacks_all": list(tc.stacks_all),
            "bb": tc.bb,
            "ante": tc.ante,
        },
    }


def _explain(result, action: str, unit: str, verb: str) -> str:
    """Feedback line. `unit` denominates the EV loss; `verb` names the oracle.

    Both are passed in rather than hardcoded, for the same reason and from the
    same defect class. The unit said "bb" for every drill kind, including ICM
    drills whose ev_loss is a $-delta ~25x larger per unit (round-3 [P7']). The
    verb said "Chart plays" for every drill, which would have claimed chart
    provenance for a solver-graded answer the moment one shipped (round-4 [10]).
    See EV_UNITS and SOURCE_VERBS.
    """
    verdict = "Correct" if result.correct else "Incorrect"
    return (f"{verdict}. {verb} {result.best_action} here — you chose "
            f"{action} (freq {result.chosen_frequency:.0%}, "
            f"EV loss {result.ev_loss_bb:.2f}{unit}).")


def _rta_json(drill: gen.Drill) -> dict:
    """The post-answer RTA panel: real solver quantities only.

    Every number is read straight from the drill's `Solution` and the
    decision-ε that actually grades it — per-action EV and chart frequency,
    the best action as EV-argmax, ε converted into the drill's own payoff
    currency via bb_value ([P7']). The reasoning sentence is server-authored
    FROM those numbers (plan §9: the page renders server words, never its own
    claims), so it can state the EV gap and the acceptance rule without the
    page inventing either.
    """
    acts = drill.solution.actions
    best = max(acts, key=lambda a: acts[a][0])
    eps = epsilon(drill.pot_bb, bb_value=drill.bb_value)
    unit = EV_UNITS[drill.kind]
    verb = SOURCE_VERBS[drill.solution.source]
    # Derived, not assumed two-action: the runner-up is the best of the rest.
    runner = max((a for a in acts if a != best), key=lambda a: acts[a][0])
    gap = acts[best][0] - acts[runner][0]
    reasoning = (
        f"{verb} {best} at {acts[best][1]:.0%} frequency — its EV beats "
        f"{runner} by {gap:.2f} {unit}. An answer grades correct when it "
        f"gives up at most ε = {eps:.2f} {unit} to the best action, or when "
        f"the chart itself mixes it at ≥{MIX_FREQ:.0%}."
    )
    if drill.kind == "icm":
        # The unit conversion is part of WHY the grade is sound, so it is
        # stated where the numbers are: EVs are prize-pool deltas and ε is
        # scaled by the average chip's pool value (scoring.epsilon).
        reasoning += (
            f" EVs here are prize-pool deltas ({unit}); 1bb of chips is worth "
            f"{drill.bb_value:.2f} of pool value, which is what scales ε."
        )
    return {
        "best_action": best,
        "epsilon": eps,
        "bb_value": drill.bb_value,
        "actions": [
            {"action": a, "ev": float(ev), "frequency": float(f)}
            for a, (ev, f) in acts.items()
        ],
        "reasoning": reasoning,
    }


def create_app(db_path: str = ":memory:", seed: int = 0) -> FastAPI:
    app = FastAPI(title="pokerlab drills")
    population = gen.default_population()
    by_id = {d.drill_id: d for d in population}
    by_cat: dict[str, list[gen.Drill]] = {}
    for d in population:
        by_cat.setdefault(d.leak_key, []).append(d)
    default_cat = population[0].leak_key

    # Unseen-rung exploration order: round-robin across kinds, NOT population
    # order. select_next serves never-drilled categories in the order the
    # caller lists them, and population order put all 630 ring categories
    # behind every HU/ICM one — a user who merged the ring feature clicked
    # through spot after spot and saw nothing new (2026-07-22 report). The
    # interleave is deterministic and only reorders EXPLORATION: due
    # categories (real leaks) still outrank every unseen one.
    by_kind: dict[str, list[str]] = {}
    seen_cats: set[str] = set()
    for d in population:
        if d.leak_key not in seen_cats:
            seen_cats.add(d.leak_key)
            by_kind.setdefault(d.kind, []).append(d.leak_key)
    cat_order: list[str] = []
    lanes = list(by_kind.values())
    for i in range(max(len(lane) for lane in lanes)):
        for lane in lanes:
            if i < len(lane):
                cat_order.append(lane[i])

    conn = db.connect(db_path, check_same_thread=False)
    lock = threading.Lock()
    rng = random.Random(seed)

    # Local single-user tool: a cached page is only ever a stale page. Two
    # separate user reports were the browser serving an old app.js after a
    # merge; the remedy is server policy, not the user's keyboard discipline.
    @app.middleware("http")
    async def _no_store(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response
    # Replay guard (round-2 finding [E43]): a double-click posts the same
    # drill_id twice and SM-2 counted both, inflating reps/interval off one
    # answer. Only the *immediately* repeated answer is absorbed — fetching the
    # next spot clears it, so genuinely re-practising a drill still counts.
    last: dict[str, object] = {"drill_id": None, "response": None}

    @app.get("/api/drill/next")
    def next_drill() -> dict:
        with lock:
            now = datetime.now(timezone.utc)
            cat = sch.select_next(conn, now, categories=cat_order) or default_cat
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
            rta = _rta_json(drill)
            if ans.action in drill.off_tree_actions:
                # OFF-TREE distractor: a real button, but outside the solved
                # game — the chart never priced it, so no ev_loss exists and
                # none is fabricated (0.0 would claim a wrong action cost
                # nothing; the gradings tier-3 rule applied here). Incorrect
                # by construction, persisted with ev_loss NULL so EV trends
                # skip it while accuracy still counts it.
                correct, ev_loss, chosen_frequency = False, None, None
                priced = "/".join(drill.legal_actions)
                explanation = (
                    f"Incorrect — off the solved tree. This drill prices only "
                    f"{priced}; {SOURCE_VERBS[drill.solution.source]} "
                    f"{rta['best_action']} here. \"{ans.action}\" may exist "
                    f"at the table, but the chart never solved it, so it has "
                    f"no EV number — it is graded wrong, not costed."
                )
            else:
                try:
                    # bb_value converts the decision-ε into the drill's own
                    # payoff currency ([P7']); 1.0 for chip-EV, the average
                    # chip value for ICM. Without it ICM was exact-argmax.
                    result = score(drill.solution, ans.action, drill.pot_bb,
                                   bb_value=drill.bb_value)
                except ValueError as exc:
                    raise HTTPException(400, str(exc)) from exc
                correct = result.correct
                ev_loss = result.ev_loss_bb
                chosen_frequency = result.chosen_frequency
                explanation = _explain(result, ans.action,
                                       EV_UNITS[drill.kind],
                                       SOURCE_VERBS[drill.solution.source])
            now = datetime.now(timezone.utc)
            db.insert_drill_attempt(conn, drill.leak_key, drill.kind, ans.action,
                                    correct, ev_loss, now.isoformat())
            sch.schedule_attempt(conn, drill.leak_key, correct, now)
            payload = {
                "drill_id": drill.drill_id,
                "correct": correct,
                # `ev_loss` + `ev_unit`, not `ev_loss_bb`: the old key asserted
                # the unit in its NAME, which was false for ICM drills. A field
                # name is a claim like any other ([P7']). None for off-tree.
                "ev_loss": ev_loss,
                "ev_unit": EV_UNITS[drill.kind],
                "best_action": rta["best_action"],
                "chosen_frequency": chosen_frequency,
                "explanation": explanation,
                # The RTA panel rides on EVERY answer; the page decides whether
                # to show it (a display toggle), never whether it exists.
                "rta": rta,
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
                # THIS session's in-flight rows: batch rows belonging to hands
                # imported in the most recent batch. `imported_at` is written
                # once per import run (hh/persist takes the caller's stamp), so
                # it is the session boundary the CLI already reports against.
                "       (SELECT COUNT(*) FROM batch_queue q"
                "          JOIN imported_hands h ON h.id = q.hand_id"
                "         WHERE q.status IN (?, ?)"
                "           AND h.imported_at ="
                "               (SELECT MAX(imported_at) FROM imported_hands)"
                "       )                                                AS queued,"
                "       (SELECT COUNT(*) FROM batch_queue"
                "         WHERE status IN (?, ?))                        AS in_flight_total",
                _IN_FLIGHT * 2,
            ).fetchone()
            session = dict(counts)
            # PARTIAL is scoped to THIS SESSION'S batch (PLAN §5.3: "until its
            # batch is solved"), which is what the CLI banner has always meant.
            # Computing it from the DB-wide count made every session inherit
            # every earlier session's backlog: a session that queued nothing
            # read as partial, and the banner told the operator to run a drain
            # that could not change one number in the report below it
            # (round-4 finding [9]).
            #
            # `in_flight_total` keeps the DB-wide view, because "is the solver
            # behind on anything at all?" is a real question and deleting the
            # count would trade a wrong answer for a missing one. It carries its
            # own scope in its own label: the defect was one WORD serving two
            # scopes, so the fix has to be two names, not a better comment.
            session["partial"] = session["queued"] > 0
            session["in_flight_label"] = IN_FLIGHT_LABEL
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
                # Shaped like `tier3`: the rows and the honesty text they
                # require travel together, so no page can render the ranking
                # while dropping what the numbers assumed (PLAN §5.3 rev-3.2 —
                # the disclosure must reach "the leak report", not stop at the
                # Solution). Unlike tier3's label this one qualifies SOME rows,
                # which is why each row carries its own count and the caveat
                # only explains what that count means.
                "leaks": {
                    "rows": views.hh_leak_report(conn, limit=5),
                    "approx_caveat": APPROX_CAVEAT,
                },
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
