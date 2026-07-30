"""Final-table ICM fixtures beyond the single bubble (plan-deferred kind).

Two new TournamentContexts join the population: FT3 (3 left, all paid —
pure ladder pressure, asymmetric stacks) and FT5 (5 left, 3 paid — the
final-table money bubble). Each gets its OWN category namespace via an
`.icm.<label>` suffix; the original BUBBLE keys stay byte-identical so no
sr_state/gradings row is orphaned (the ante=0 rule, third application).
"""

import pytest

from pokerlab.charts import hands
from pokerlab.drills.categories import jamfold_category
from pokerlab.drills.generator import FT3, FT5, icm_drills


def test_categories_suffix_and_bubble_keys_unchanged():
    assert (jamfold_category("SB", 8.0, icm=True, icm_label="ft3")
            == "SBjam.icm.ft3|preflop|jam|8")
    # the pin that protects existing rows: bare icm keys are byte-identical
    assert (jamfold_category("SB", 10.0, icm=True)
            == "SBjam.icm|preflop|jam|10")


@pytest.fixture(scope="module")
def ft3():
    return icm_drills(FT3, sb_seat=2, bb_seat=0, label="ft3")


def test_ft3_drills_have_their_own_keys_and_context(ft3):
    d = next(x for x in ft3 if x.position == "SB" and x.hand_label == "AA")
    assert ".icm.ft3|" in d.leak_key
    assert d.kind == "icm"
    assert d.tournament is FT3
    # all three paid -> this is final-table ladder pressure, not a bubble,
    # and the prompt must say so (the ladder is part of the question)
    assert "Final-table ICM" in d.description
    assert "3 left, 3 paid" in d.description


def test_ft3_bb_value_is_pool_over_total_chips(ft3):
    d = ft3[0]
    # pool 1000 over (1200+1000+800)/100 = 30bb of chips -> 1bb = 33.33$
    assert d.bb_value == pytest.approx(1000.0 / 30.0)


def test_ft3_aa_always_continues(ft3):
    for pos in ("SB", "BB"):
        d = next(x for x in ft3 if x.position == pos and x.hand_label == "AA")
        best = max(d.solution.actions, key=lambda a: d.solution.actions[a][0])
        assert best in ("jam", "call")
        assert d.solution.actions[best][1] > 0.95


def test_ft5_is_a_bubble_and_in_population():
    ft5 = icm_drills(FT5, sb_seat=0, bb_seat=1, label="ft5")
    d = ft5[0]
    assert ".icm.ft5|" in d.leak_key
    assert "Bubble ICM" in d.description        # 5 left, 3 paid IS a bubble
    from pokerlab.drills.generator import default_population
    keys = {x.leak_key for x in default_population()}
    assert any(".icm.ft3|" in k for k in keys)
    assert any(".icm.ft5|" in k for k in keys)
