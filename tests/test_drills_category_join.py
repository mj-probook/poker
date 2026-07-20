"""Integration contract: the leak→drill→SM-2 loop joins on ONE canonical key.

A known HH leak (SB open-fold at push/fold depth) must produce a
gradings.leak_key that is byte-identical to (a) the drill the generator emits
for that spot, (b) a row in the drill-attempt leak report, and (c) the category
SM-2 resurfaces next. This is the F↔E seam Slice I's end-to-end smoke extends.
"""

from datetime import datetime
from pathlib import Path

from pokerlab.charts.jamfold import jamfold_range
from pokerlab.drills.categories import jamfold_category, snap_depth
from pokerlab.drills.generator import jamfold_drills
from pokerlab.drills.scheduler import schedule_attempt, select_next
from pokerlab.hh.decisions import Decision, extract_decisions
from pokerlab.hh.ggpoker import parse_ggpoker
from pokerlab.hh.population import load_population
from pokerlab.hh.persist import persist_session
from pokerlab.hh.report import grade_session
from pokerlab.store import db
from pokerlab.store.views import hh_leak_report, worst_leak_categories
from pokerlab.types import TIER_CHART

FIXTURES = Path(__file__).parent / "fixtures" / "hh"
AT = "2026-07-19T00:00:00"
_AA = (48, 49)   # As Ad -> hand class "AA"
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

    # 2. the drill generator emits a drill with EXACTLY that leak_key.
    assert key in {dr.leak_key for dr in jamfold_drills()}

    # 3. persisted, the HH leak surfaces in the tiers-1/2 leak report...
    conn = db.connect()
    persist_session(conn, [parsed], report, graded_at=AT, raw_texts=[raw])
    assert any(r["leak_key"] == key for r in hh_leak_report(conn))

    # ...and drilling that same category shows up in the drill-attempt report.
    for ok in (0, 0, 1):
        db.insert_drill_attempt(conn, key, "jamfold", "fold", bool(ok), 0.0, AT)
    assert any(r["leak_key"] == key for r in worst_leak_categories(conn))

    # 4. SM-2 schedules the category and next-due selection picks it up.
    schedule_attempt(conn, key, correct=False, now=NOW)
    assert select_next(conn, NOW) == key


# --------------------------------------------------------------------------- #
# Round-2 finding [E33]: the seam above joined on the KEY but nothing asserted
# the two sides carried the same ANSWER. The leak_key was built from the snapped
# bucket while the grader solved the chart at the RAW eff_bb, so a hand graded
# at 12.5bb was scored against a chart the drill for that category never uses.
# Every HH fixture happens to sit exactly on a bucket, which is why the string
# join looked healthy. Solution equality is the invariant the seam claims.
# --------------------------------------------------------------------------- #
def _decision(pos: str, eff_bb: float, hand: tuple[int, int], chosen):
    from pokerlab.types import TIER_CHART as T
    return Decision(
        index=0, street="preflop", seat=0, position=pos, num_in_pot=2,
        pot=0, pot_bb=2.0 * eff_bb, eff_bb=eff_bb, ante_bb=0.0, to_call=0,
        opp_allin=(pos == "BB"), hole=hand, board=(), legal=[], chosen=chosen,
        is_allin=(pos == "SB"), tier=T, formation="2max:" + pos,
        action_type="jam" if pos == "SB" else "call",
    )


def test_grader_and_drill_agree_on_the_answer_key_off_bucket() -> None:
    """At an off-bucket depth the grader must use the drill's chart, not its own."""
    from pokerlab.charts.hands import HAND_CLASSES
    from pokerlab.hh.grade import grade_tier1

    from pokerlab.charts.hands import card_combos

    by_id = {dr.drill_id: dr for dr in jamfold_drills()}
    off_bucket = 12.5                       # snaps to the 10bb bucket
    best = lambda s: max(s.actions, key=lambda a: s.actions[a][0])  # noqa: E731

    # Sweep EVERY hand class, not a convenient one: AA jams at both depths, so
    # a single-hand assertion here passes even with the bug reverted. The flips
    # live in the marginal classes (J3s, T5s, K4s, ...).
    for pos, act in (("SB", ("allin", 0)), ("BB", ("call", 0))):
        for hand in HAND_CLASSES:
            hole = card_combos(hand)[0]
            g = grade_tier1(_decision(pos, off_bucket, hole, act))
            drill = by_id[f"{g.leak_key}:{hand}"]
            assert g.best == best(drill.solution), (
                f"{pos} {hand}: grader says {g.best}, "
                f"drill {g.leak_key} teaches {best(drill.solution)}")


def test_raw_and_snapped_charts_really_do_differ_at_this_depth() -> None:
    """Guards the test above from going vacuous if snapping ever becomes a no-op."""
    from pokerlab.charts.hands import HAND_CLASSES

    flips = 0
    for pos in ("SB", "BB"):
        raw = jamfold_range(pos, 12.5)
        snapped = jamfold_range(pos, float(snap_depth(12.5)))
        for hand in HAND_CLASSES:
            best = lambda s: max(s.actions, key=lambda a: s.actions[a][0])  # noqa: E731
            flips += best(raw[hand]) != best(snapped[hand])
    assert flips > 0, "raw and snapped charts agree — E33 fixture is vacuous"
