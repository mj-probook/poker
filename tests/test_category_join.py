"""Integration contract: the leak→drill→SM-2 loop joins on ONE canonical key.

A known HH leak (SB open-fold at push/fold depth) must produce a
gradings.leak_key that is byte-identical to (a) the drill the generator emits
for that spot, (b) a row in the drill-attempt leak report, and (c) the category
SM-2 resurfaces next. This is the F↔E seam Slice I's end-to-end smoke extends.
"""

from datetime import datetime
from pathlib import Path

from pokerlab.drills.categories import jamfold_category
from pokerlab.drills.generator import jamfold_drills
from pokerlab.drills.scheduler import schedule_attempt, select_next
from pokerlab.hh.decisions import extract_decisions
from pokerlab.hh.ggpoker import parse_ggpoker
from pokerlab.hh.population import load_population
from pokerlab.hh.persist import persist_session
from pokerlab.hh.report import grade_session
from pokerlab.store import db
from pokerlab.store.views import hh_leak_report, worst_leak_categories
from pokerlab.types import TIER_CHART

FIXTURES = Path(__file__).parent / "fixtures" / "hh"
AT = "2026-07-19T00:00:00"
NOW = datetime.fromisoformat(AT)


def test_hh_leak_joins_drill_generator_and_scheduler() -> None:
    raw = (FIXTURES / "gg_sb_fold_leak.txt").read_text()
    parsed = parse_ggpoker(raw)

    # 1. HH grading of the SB open-fold leak -> a canonical leak_key.
    report = grade_session([parsed], population=load_population())
    leak = next(gd.grading for gd in report.graded
                if gd.grading.tier == TIER_CHART and gd.grading.graded
                and (gd.grading.ev_loss or 0) > 1.0)
    key = leak.leak_key
    d = extract_decisions(parsed)[0]
    assert key == jamfold_category("SB", d.eff_bb)  # depth-bucketed spot

    # 2. the drill generator emits a drill with EXACTLY that spot_key.
    assert key in {dr.spot_key for dr in jamfold_drills()}

    # 3. persisted, the HH leak surfaces in the tiers-1/2 leak report...
    conn = db.connect()
    persist_session(conn, [parsed], report, graded_at=AT, raw_texts=[raw])
    assert any(r["leak_key"] == key for r in hh_leak_report(conn))

    # ...and drilling that same category shows up in the drill-attempt report.
    for ok in (0, 0, 1):
        db.insert_drill_attempt(conn, key, "jamfold", "fold", bool(ok), 0.0, AT)
    assert any(r["spot_key"] == key for r in worst_leak_categories(conn))

    # 4. SM-2 schedules the category and next-due selection picks it up.
    schedule_attempt(conn, key, correct=False, now=NOW)
    assert select_next(conn, NOW) == key
