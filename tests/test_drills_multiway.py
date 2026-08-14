"""Multiway postflop drills — tier 3, the honesty-critical kind.

The load-bearing rule (CLAUDE.md, plan §5.3): multiway spots NEVER report
ev_loss — no trustworthy multiway EV reference exists, so the only honest
grading is the frequency-deviation contract `hh.grade_tier3` already ships:
compare the chosen action TYPE to the versioned population table, flag rare
actions, claim nothing else. No correct/incorrect, no EV.

What the built machinery CAN honestly add is ADVISORY analysis, clearly
labeled: a CERTIFIED HU-collapsed river solve (both villains merged into one
uniform-range player — the tier-2 approximation contract, stated) and
seeded Monte Carlo equity vs the field. Advisory informs study; it never
grades.
"""

import pytest

from pokerlab.drills.multiway import (MW_TABLE, load_multiway_advisory,
                                      multiway_drills)
from pokerlab.types import TIER_BEST_AVAILABLE


def test_multiway_drills_exist_for_flop_and_river():
    drills = multiway_drills()
    assert drills
    streets = {d.leak_key.split("|")[1] for d in drills}
    assert streets == {"flop", "river"}
    assert all(d.kind == "multiway" for d in drills)
    assert all(d.tier == TIER_BEST_AVAILABLE for d in drills)


def test_multiway_never_carries_an_ev_number():
    """The hard rule, enforced at the source: tier-3 Solutions carry NO
    action EVs at all — nothing downstream can leak a number that does not
    exist."""
    for d in multiway_drills():
        assert d.solution.actions == {}, d.drill_id
        assert d.population_key == "6max:BB"
        assert d.legal_actions == ("check", "bet")   # action TYPES (tier 3)


def test_multiway_scene_is_three_handed_with_a_real_board():
    d = multiway_drills()[0]
    assert d.table == MW_TABLE == ("BB", "CO", "BTN")
    assert len(d.board) in (3, 5)
    assert len(d.hero_cards) == 2
    assert not set(d.hero_cards) & set(d.board)
    assert d.pot_bb == 8.0


def test_advisory_solve_recertifies_and_is_disclosed():
    """River advisory solves ship with the same certificate as every other
    artifact: rebuild the tree, inject stored strategies, MEASURE the gap."""
    solves = load_multiway_advisory()
    assert solves
    for board, (solver, avg, gap) in solves.items():
        assert gap <= 0.01, (board, gap)
        assert solver.exploitability(avg=avg) <= 0.015, board
    river = [d for d in multiway_drills() if len(d.board) == 5]
    d = river[0]
    adv = d.advisory
    assert adv is not None
    assert "advisory" in adv["caveat"].lower()
    assert "uniform" in adv["caveat"]
    assert set(adv["solve"]) >= {"check", "jam"}     # per-action bb EVs
    # equity is seeded MC, disclosed as such
    assert 0.0 <= adv["equity_vs_field"] <= 1.0
    assert "seed" in adv["equity_note"]


def test_flop_drills_carry_equity_advisory_but_no_solve():
    flop = [d for d in multiway_drills() if len(d.board) == 3]
    d = flop[0]
    assert d.advisory is not None
    assert "solve" not in d.advisory     # a flop solve would be fabricated
    assert 0.0 <= d.advisory["equity_vs_field"] <= 1.0


def test_aa_has_more_equity_than_72o_on_neutral_boards():
    """Sanity anchor on the MC equity: AA beats 72o averaged across the
    shipped flop boards (any single board can flip; the average cannot)."""
    flop = [d for d in multiway_drills() if len(d.board) == 3]
    aa = [d.advisory["equity_vs_field"] for d in flop if d.hand_label == "AA"]
    trash = [d.advisory["equity_vs_field"] for d in flop
             if d.hand_label == "72o"]
    assert aa and trash
    assert sum(aa) / len(aa) > sum(trash) / len(trash)


def test_missing_artifact_fails_with_the_regen_command(tmp_path):
    """BOTH loaders name the gen script when the npz is absent. The builder
    calls load_equities first, so a guard only on load_multiway_advisory is
    unreachable — the user would get numpy's bare error instead of the fix
    (night-shift review [1])."""
    from pokerlab.drills.multiway import load_equities
    gone = str(tmp_path / "nope.npz")
    for loader in (load_equities, load_multiway_advisory):
        with pytest.raises(FileNotFoundError, match="gen_multiway_advisory"):
            loader(gone)


def test_board_drift_names_the_regen_command():
    """A board list that drifted from the shipped artifact must say how to
    fix it, not die with a bare KeyError (night-shift review [1])."""
    from pokerlab.drills.multiway import _require_board
    with pytest.raises(KeyError, match="gen_multiway_advisory"):
        _require_board({"2c2d2h2s3c": object()}, "AhKhQh7d2s")
