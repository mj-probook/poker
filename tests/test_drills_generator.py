"""Slice E — drill generator from the Slice-C chart engine (plan §5.1).

Answer keys are chart-derived only. The generator is proven to (a) cover the
5–20bb × position syllabus, (b) faithfully carry the chart Solution, (c) build a
bubble ICM set that is genuinely tighter than the chip-EV range, and (d) produce
drills on which a wrong action actually fails scoring.
"""

import pytest

from pokerlab.charts import hands, jamfold_range
from pokerlab.drills import generator as gen
from pokerlab.drills.scoring import score


def test_jamfold_population_covers_syllabus():
    drills = gen.jamfold_drills()
    assert len(drills) == 169 * 2 * len(gen.DEPTHS)
    # one category per (position, depth)
    assert len({d.spot_key for d in drills}) == 2 * len(gen.DEPTHS)
    sb = [d for d in drills if d.position == "SB"]
    assert all(d.legal_actions == ("jam", "fold") for d in sb)
    assert all(d.pot_bb == 2.0 * d.depth_bb for d in drills)
    assert all(d.drill_id == f"{d.spot_key}:{d.hand_label}" for d in drills)


def test_jamfold_drill_carries_chart_solution():
    drills = {d.drill_id: d for d in gen.jamfold_drills()}
    d = drills["SBjam:10bb:AKs"]
    assert d.kind == "jamfold" and d.tournament is None
    assert d.solution.source == "chart"
    assert d.solution.actions == jamfold_range("SB", 10)["AKs"].actions


def test_icm_drills_carry_tournament_context():
    drills = gen.icm_drills()
    assert len(drills) == 169 * 2
    assert all(d.tournament is gen.BUBBLE for d in drills)
    assert all(d.kind == "icm" for d in drills)
    assert all(d.depth_bb == 10.0 for d in drills)         # 1000 chips / 100 bb
    assert all("Bubble ICM" in d.description for d in drills)


def test_icm_bb_range_is_tighter_than_chip_ev():
    icm = {d.hand_label: d for d in gen.icm_drills() if d.position == "BB"}
    chip = jamfold_range("BB", 10)
    icm_call = sum(icm[h].solution.actions["call"][1] * hands.combos(h)
                   for h in hands.HAND_CLASSES)
    chip_call = sum(chip[h].actions["call"][1] * hands.combos(h)
                    for h in hands.HAND_CLASSES)
    assert icm_call < chip_call            # bubble risk premium tightens calls
    # ...but never folds aces
    assert icm["AA"].solution.actions["call"][1] >= 0.99


def test_icm_wrong_action_fails_scoring():
    # AA jams under ICM too; folding it must grade incorrect (freq ~0, big $ loss).
    aa = next(d for d in gen.icm_drills()
              if d.position == "SB" and d.hand_label == "AA")
    s = score(aa.solution, "fold", aa.pot_bb)
    assert not s.correct and s.best_action == "jam"


def test_sample_drills_is_deterministic_and_mixed():
    pool = gen.jamfold_drills()
    a = gen.sample_drills(pool, 500, seed=0)
    b = gen.sample_drills(pool, 500, seed=0)
    assert [d.drill_id for d in a] == [d.drill_id for d in b]
    assert len(a) == 500
    # a 500-sample of a 1690 pool spans both positions and several depths
    assert len({d.position for d in a}) == 2
    assert len({d.depth_bb for d in a}) == len(gen.DEPTHS)
