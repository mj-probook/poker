"""HH grader table-size keying (the seam ring.py's docstring recorded).

Since the ring charts existed only at 9-max, the grader keyed EVERY folded-
to-SB hand to the HU `SBjam` chart — pricing 2 antes of dead money on a
hand that had 9. The table-size axis solved every size 3–9, so the grader
now keys the SB first-in decision to the ring chart at the hand's ACTUAL
table size (Decision.formation carries it). HU hands stay byte-identical.

The BB-defend half of the seam stays HU-keyed FOR A STATED REASON: the ring
defense key needs the JAMMER's position, which Decision does not carry yet
— the grading note says so on every affected hand, so the approximation is
disclosed per-hand, never silent.
"""

from pokerlab.hh.decisions import Decision
from pokerlab.hh.grade import grade_tier1
from pokerlab.types import TIER_CHART


def _sb_first_in(formation: str, num_in_pot: int = 2) -> Decision:
    return Decision(
        index=0, street="preflop", seat=0, position="SB",
        num_in_pot=num_in_pot, pot=30, pot_bb=1.5, eff_bb=10.0, ante_bb=0.0,
        to_call=10, opp_allin=False, hole=(48, 49),   # ~AA-ish cards
        board=(), legal=[("fold", 0)], chosen=("allin", 200), is_allin=True,
        tier=TIER_CHART, formation=formation, action_type="jam",
    )


def test_nine_max_sb_jam_keys_to_the_ring_chart():
    g = grade_tier1(_sb_first_in("9max:SB"))
    assert g.graded is True
    assert g.leak_key == "SBjam.9max|preflop|jam|10"
    assert "ring9" in g.provenance          # the ring chart's own range_ctx


def test_six_max_sb_jam_keys_to_the_six_max_chart():
    g = grade_tier1(_sb_first_in("6max:SB"))
    assert g.leak_key == "SBjam.6max|preflop|jam|10"
    assert "ring6" in g.provenance


def test_heads_up_sb_stays_byte_identical():
    g = grade_tier1(_sb_first_in("2max:SB"))
    assert g.leak_key == "SBjam|preflop|jam|10"
    assert "jamfold" in g.provenance


def test_bb_defend_at_nine_max_discloses_the_hu_keying():
    d = Decision(
        index=0, street="preflop", seat=1, position="BB", num_in_pot=2,
        pot=210, pot_bb=10.5, eff_bb=10.0, ante_bb=0.0, to_call=190,
        opp_allin=True, hole=(48, 49), board=(), legal=[("fold", 0)],
        chosen=("call", 190), is_allin=True, tier=TIER_CHART,
        formation="9max:BB", action_type="call",
    )
    g = grade_tier1(d)
    assert g.graded is True
    assert g.leak_key == "BBcall|preflop|call|10"    # still the HU key
    assert "HU chart" in g.note                     # ...and it SAYS so
