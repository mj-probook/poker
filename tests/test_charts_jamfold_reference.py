"""Slice C — jam/fold reference-table checks (impl doc §3 Slice C exit).

Two DIFFERENT things live here, and conflating them overstated the evidence
(round-1 finding [3]):

1. **Regression guard vs a frozen self-generated table** — the ≥99% agreement
   check. The baseline in tests/fixtures/jamfold_reference.json is *our own
   solver's* output, frozen. Agreement with it proves the solver has not
   drifted; it is NOT a check against published Nash charts, and never was.
   Regenerate with scripts/gen_jamfold_reference.py only on an intentional
   model change.

2. **Hand-audited published-fact check** — the ~30 entries in
   `_meta.hand_audited`, each transcribed from a well-known published HU
   push/fold fact into a machine-readable per-depth expected action. These
   assert the SOLVER's action, so they are the part that can actually catch the
   solver being wrong rather than merely inconsistent. (Nothing vendor-derived
   enters the table — the facts are a sanity cross-reference only, per
   CLAUDE.md.)
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
    """Regression guard vs the FROZEN SELF-GENERATED table — drift detection.

    Not a published-Nash check: the baseline is our own solver's frozen output,
    so this can only catch the solver changing, never the solver being wrong.
    The published-fact evidence is the hand-audited test below.
    """
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
    depths = {str(d) for d in ref["_meta"]["depths_bb"]}
    for entry in audited:
        assert entry["hand"] in hands.HAND_INDEX
        assert entry["position"] in ("SB", "BB")
        assert entry["published_fact"]
        # every entry must state a checkable direction, or it is decorative
        assert entry["expect"], f"{entry['hand']} has no audited expectation"
        assert set(entry["expect"]) <= depths
        legal = {"jam", "fold"} if entry["position"] == "SB" else {"call", "fold"}
        assert set(entry["expect"].values()) <= legal


def test_solver_matches_every_hand_audited_published_fact(ref):
    """[3] The audit is load-bearing: the SOLVER must take each stated action.

    This is the only check here that compares against something other than our
    own frozen output, so it is what would catch a solver that is consistently
    wrong rather than merely drifting.
    """
    failures = []
    for entry in ref["_meta"]["hand_audited"]:
        i = hands.HAND_INDEX[entry["hand"]]
        sb = entry["position"] == "SB"
        for depth, expected in entry["expect"].items():
            sol = solve_jamfold(float(depth))     # lru_cached across depths
            freq = float(sol.sb_jam[i] if sb else sol.bb_call[i])
            actual = ("jam" if sb else "call") if _action(freq) else "fold"
            if actual != expected:
                failures.append(
                    f"{entry['position']} {entry['hand']} @{depth}bb: "
                    f"solver={actual} (freq {freq:.3f}) but audited as "
                    f"{expected} — {entry['published_fact']}")
    assert not failures, "audited published facts violated:\n" + "\n".join(failures)


def test_audited_premium_and_trash_entries_are_consistent(ref):
    # Spot-check that the frozen table matches the audited-fact direction.
    assert ref["sb_jam"]["10"]["AA"] >= 0.99      # premium jams
    assert ref["sb_jam"]["15"]["72o"] < 0.5       # trash folds
    assert ref["bb_call"]["10"]["AA"] >= 0.99     # premium calls
    assert ref["bb_call"]["10"]["72o"] < 0.5      # trash never calls
