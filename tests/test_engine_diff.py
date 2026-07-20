"""Slice A M0 exit: differential vs PokerKit on seeded random hands.

Generates random NLHE hands, plays them through our engine, replays the
identical hand in PokerKit, and asserts identical final per-seat stacks
(pot/side-pot/showdown mechanics). PokerKit is the presumed-correct oracle.

Documented quirk (see diff_harness._is_odd_chip_only): in rare multiway all-in
ties PokerKit places the single odd chip of a split on a different tied winner
than our standard "lowest eligible seat" rule — a labelling difference, not a
mechanics bug (winners, pot totals and stacks all agree up to that one chip
among identically-ranked winners). These are counted separately; the test fails
on any *substantive* disagreement.
"""

import pytest

from tests.diff_harness import bb_ante_setup, run_differential, short_stack_setup

# ~0.2% of random hands hit the odd-chip-placement quirk; keep a safety ceiling.
_QUIRK_CEILING = 0.01


def test_differential_vs_pokerkit_20k() -> None:
    exact, oddchip, mismatches = run_differential(20_000)
    total = exact + oddchip + len(mismatches)
    assert total == 20_000
    assert not mismatches, (
        f"{len(mismatches)} substantive PokerKit disagreements (bugs):\n"
        + "\n".join(mismatches[:5])
    )
    assert oddchip <= total * _QUIRK_CEILING, (
        f"odd-chip-placement quirk fired {oddchip}/{total} times — "
        "unexpectedly high, investigate for a real side-pot bug"
    )


def test_differential_vs_pokerkit_short_stacks_20k() -> None:
    """Wave-2 [E1]: the same oracle over hands with sub-blind stacks.

    `random_setup` floors every stack at 2bb, so it never generates a hand where
    a blind or ante puts a player all-in before anyone acts. That blind spot hid
    a real close-out bug: the engine kept offering a check to a player whose only
    live opponent was already all-in. The 20k baseline above was green throughout
    (0/4000 on a targeted probe); this generator caught it at 312/4000 (7.8%).

    A player short of a blind is routine in MTTs, so this generator is pinned as
    a permanent second axis of the M0 exit rather than a one-off probe.
    """
    exact, oddchip, mismatches = run_differential(20_000, setup_fn=short_stack_setup)
    total = exact + oddchip + len(mismatches)
    assert total == 20_000
    assert not mismatches, (
        f"{len(mismatches)} substantive PokerKit disagreements (bugs):\n"
        + "\n".join(mismatches[:5])
    )
    assert oddchip <= total * _QUIRK_CEILING


def test_differential_vs_pokerkit_bb_ante_20k() -> None:
    """Wave-3 [A2]: the same oracle over BIG-BLIND-ANTE hands.

    The other two generators emit only uniform antes or none, so neither can
    produce a single-payer dead-money contribution — the shape wave-3 [A1] lived
    in, where the engine returned a losing BB's own ante to it as though it were
    an uncalled bet. Both existing axes were green through all of it.

    BB ante is the standard modern MTT structure, so this is the common case.
    Pinned as a permanent third axis for the same reason as `short_stack_setup`:
    a differential is only as good as its generator's support.
    """
    exact, oddchip, mismatches = run_differential(20_000, setup_fn=bb_ante_setup)
    total = exact + oddchip + len(mismatches)
    assert total == 20_000
    assert not mismatches, (
        f"{len(mismatches)} substantive PokerKit disagreements (bugs):\n"
        + "\n".join(mismatches[:5])
    )
    assert oddchip <= total * _QUIRK_CEILING


@pytest.mark.slow
def test_differential_vs_pokerkit_1m() -> None:
    exact, oddchip, mismatches = run_differential(1_000_000)
    total = exact + oddchip + len(mismatches)
    assert total == 1_000_000
    assert not mismatches, (
        f"{len(mismatches)} substantive PokerKit disagreements (bugs):\n"
        + "\n".join(mismatches[:10])
    )
    assert oddchip <= total * _QUIRK_CEILING


@pytest.mark.slow
def test_differential_vs_pokerkit_bb_ante_1m() -> None:
    exact, oddchip, mismatches = run_differential(1_000_000, setup_fn=bb_ante_setup)
    total = exact + oddchip + len(mismatches)
    assert total == 1_000_000
    assert not mismatches, (
        f"{len(mismatches)} substantive PokerKit disagreements (bugs):\n"
        + "\n".join(mismatches[:10])
    )
    assert oddchip <= total * _QUIRK_CEILING


@pytest.mark.slow
def test_differential_vs_pokerkit_short_stacks_1m() -> None:
    exact, oddchip, mismatches = run_differential(1_000_000, setup_fn=short_stack_setup)
    total = exact + oddchip + len(mismatches)
    assert total == 1_000_000
    assert not mismatches, (
        f"{len(mismatches)} substantive PokerKit disagreements (bugs):\n"
        + "\n".join(mismatches[:10])
    )
    assert oddchip <= total * _QUIRK_CEILING
