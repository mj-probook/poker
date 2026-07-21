"""Ring drill population: 9-max first-in jams + every defend-vs-jam pair.

Categories follow the canonical 4-part key with new formations owned by
drills/categories.py: `COjam|preflop|jam|10`, `BBcall.vUTG|preflop|call|10`.
Drills carry the formation facts the table scene needs (`table`, `versus`) so
the web layer can draw seats from server facts instead of inferring them.
"""

import pytest

from pokerlab.charts.ring import RING_JAMMERS, RING_ORDER
from pokerlab.drills.categories import ring_category
from pokerlab.drills.generator import jamfold_drills, ring_drills


def test_ring_category_keys():
    assert ring_category("CO", 10.0) == "COjam|preflop|jam|10"
    assert (ring_category("BB", 10.0, versus="UTG", ante_bb=0.125)
            == "BBcall.vUTG|preflop|call|10a0.125")


@pytest.fixture(scope="module")
def drills():
    return ring_drills()


def test_ring_population_covers_jammers_and_all_defender_pairs(drills):
    pairs = sum(len(RING_ORDER) - 1 - RING_ORDER.index(j)
                for j in RING_JAMMERS)          # 35 defender pairs
    assert len(drills) == (len(RING_JAMMERS) + pairs) * 5 * 3 * 169
    assert {d.kind for d in drills} == {"ring"}


def test_ring_jam_drill_shape(drills):
    d = next(x for x in drills if x.position == "CO" and not x.versus
             and x.hand_label == "AKs" and x.depth_bb == 10.0
             and "a" not in x.leak_key.rsplit("|", 1)[1])
    assert d.legal_actions == ("jam", "fold")
    assert len(d.off_tree_actions) >= 3        # first-in spots get distractors
    assert d.action_note == ""
    assert d.table == RING_ORDER
    assert "jam" not in d.description.lower()  # prompt must not leak the pair


def test_ring_defense_drill_shape(drills):
    d = next(x for x in drills if x.position == "BB" and x.versus == "CO"
             and x.hand_label == "AA" and x.depth_bb == 10.0)
    assert d.legal_actions == ("call", "fold")
    assert d.off_tree_actions == ()            # facing an all-in: complete at 2
    assert d.action_note
    assert "CO all-in" in d.description
    assert d.table == RING_ORDER


def test_hu_drills_name_their_jammer():
    hu = jamfold_drills(depths=(10,))
    bb = next(d for d in hu if d.position == "BB")
    sb = next(d for d in hu if d.position == "SB")
    assert bb.versus == "SB"                   # the jam the BB is facing
    assert sb.versus == ""                     # first to act, nobody jammed
