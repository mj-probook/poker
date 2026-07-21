"""The table scene must SHOW the action, not only narrate it under the felt.

User-reported defect: a BB drill is a decision about an all-in the scene never
drew — the jam lived only in the description sentence below the table, which
reads as flavor text. The payload now carries the prior action as
server-authored fields (`action_line`, `facing_allin`) so the page can draw it
without deriving its own account of what happened.

Second, coupled fix: now that the button row contains off-tree distractors, an
SB prompt that enumerated "open-jam or fold?" would identify the real pair and
defeat them, so the SB description must not name the priced actions. The BB
prompt keeps "call or fold?" — facing an all-in that pair is poker-complete,
which is also why BB spots carry no distractors.
"""

import pytest

from pokerlab.drills import generator as gen
from pokerlab.web.app import _spot_json


@pytest.fixture(scope="module")
def population():
    return gen.default_population()


def test_bb_spot_states_the_allin_on_the_table(population):
    bb = next(d for d in population if d.position == "BB")
    spot = _spot_json(bb)
    assert spot["facing_allin"] is True
    assert "all-in" in spot["action_line"]


def test_sb_spot_states_action_on_hero(population):
    sb = next(d for d in population if d.position == "SB")
    spot = _spot_json(sb)
    assert spot["facing_allin"] is False
    assert spot["action_line"]          # non-empty: the scene always shows it


def test_sb_description_does_not_name_the_priced_pair(population):
    sb = next(d for d in population if d.position == "SB")
    assert "jam" not in _spot_json(sb)["description"].lower()
