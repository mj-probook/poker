"""Slice I item 4: end-to-end M4/M2 loop smoke (the whole system in one test).

Import a fixture HH session → grade every tier (1 chart, 2 real solver, 3
population) → persist → leak report → SM-2 registers the top leak → the web API
serves a drill for that exact category and scores an answer. One slow test; it
is the integration contract the whole build converges on.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pokerlab.hh.decisions import extract_decisions
from pokerlab.hh.ggpoker import parse_ggpoker
from pokerlab.hh.persist import persist_session
from pokerlab.hh.pokerstars import parse_pokerstars
from pokerlab.hh.population import load_population
from pokerlab.hh.report import grade_session
from pokerlab.solver import subgame as sg
from pokerlab.spike import tier2
from pokerlab.store import db
from pokerlab.store.views import hh_leak_report
from pokerlab.drills.scheduler import schedule_attempt, select_next
from pokerlab.web.app import create_app

FIXTURES = Path(__file__).parent / "fixtures" / "hh"
AT = "2026-07-19T00:00:00"
FAST_CFG = sg.BetConfig(sizes=(0.75,), jam=False, max_raises=0)


@pytest.mark.slow
def test_full_loop_hh_to_web_drill(tmp_path):
    dbp = str(tmp_path / "e2e.db")
    conn = db.connect(dbp)

    # 1. import a session spanning all three tiers.
    files = ["gg_sb_fold_leak.txt", "ps_ante_hu.txt", "ps_multiway_flop.txt"]
    parsers = [parse_ggpoker, parse_pokerstars, parse_pokerstars]
    raws = [(FIXTURES / f).read_text() for f in files]
    parsed = [p(r) for p, r in zip(parsers, raws)]

    # 2. pre-cache the HU river solve so tier-2 grades inline via the real solver.
    river = extract_decisions(parsed[1])[4]
    assert tier2.cache_solve(conn, river, iters=80, cfg=FAST_CFG) is not None

    # 3. grade tiers 1+2+3 and persist.
    report = grade_session(parsed, population=load_population(),
                           solution_for=tier2.make_inline_solution_for(conn))
    tiers = {gd.grading.tier for gd in report.graded if gd.grading.graded}
    assert tiers == {1, 2, 3}                       # all three tiers really fired
    persist_session(conn, parsed, report, graded_at=AT, raw_texts=raws)

    # 4. leak report -> top leak; register it with SM-2 (the "resurface" step).
    leaks = hh_leak_report(conn, limit=5)
    top = leaks[0]["leak_key"]
    assert top == "SBjam|preflop|jam|10"
    now = datetime.now(timezone.utc)
    schedule_attempt(conn, top, correct=False, now=now)
    assert select_next(conn, now) == top

    # 5. the web API serves a drill for that category and scores an answer.
    app = create_app(db_path=dbp, seed=0)
    with TestClient(app) as client:
        spot = client.get("/api/drill/next").json()
        assert spot["drill_id"].startswith(top + ":")   # a drill for the top leak
        answer = client.post("/api/drill/answer",
                             json={"drill_id": spot["drill_id"], "action": "jam"}).json()
        assert "correct" in answer and "ev_loss_bb" in answer
        assert answer["best_action"] in {"jam", "fold"}
