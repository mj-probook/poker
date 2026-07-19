"""Slice C — jam/fold reference-table regression (impl doc §3 Slice C exit).

Guards the solver against drift: the current solve must still agree with the
frozen tests/fixtures/jamfold_reference.json on ≥99% of hand decisions at every
audited depth. Regenerate the baseline with scripts/gen_jamfold_reference.py
only on an intentional model change.
"""

import json
from pathlib import Path

import pytest

from pokerlab.charts import hands
from pokerlab.charts.jamfold import solve_jamfold

FIXTURE = Path(__file__).parent / "fixtures" / "jamfold_reference.json"


@pytest.fixture(scope="module")
def ref():
    return json.loads(FIXTURE.read_text())


def _action(freq: float) -> bool:
    return freq >= 0.5


def test_solver_agrees_with_frozen_reference_99pct(ref):
    match = total = 0
    for depth in ref["_meta"]["depths_bb"]:
        sol = solve_jamfold(float(depth))
        sb = ref["sb_jam"][str(depth)]
        bb = ref["bb_call"][str(depth)]
        for i, h in enumerate(hands.HAND_CLASSES):
            total += 2
            match += _action(sol.sb_jam[i]) == _action(sb[h])
            match += _action(sol.bb_call[i]) == _action(bb[h])
    assert match / total >= 0.99, f"agreement {match}/{total} = {match/total:.4f}"


def test_reference_covers_all_169_hands_at_every_depth(ref):
    for depth in ref["_meta"]["depths_bb"]:
        assert set(ref["sb_jam"][str(depth)]) == set(hands.HAND_CLASSES)
        assert set(ref["bb_call"][str(depth)]) == set(hands.HAND_CLASSES)


def test_at_least_30_hand_audited_entries_documented(ref):
    audited = ref["_meta"]["hand_audited"]
    assert len(audited) >= 30
    for entry in audited:
        assert entry["hand"] in hands.HAND_INDEX
        assert entry["position"] in ("SB", "BB")
        assert entry["published_fact"]


def test_audited_premium_and_trash_entries_are_consistent(ref):
    # Spot-check that the frozen table matches the audited-fact direction.
    assert ref["sb_jam"]["10"]["AA"] >= 0.99      # premium jams
    assert ref["sb_jam"]["15"]["72o"] < 0.5       # trash folds
    assert ref["bb_call"]["10"]["AA"] >= 0.99     # premium calls
    assert ref["bb_call"]["10"]["72o"] < 0.5      # trash never calls
