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

from tests.diff_harness import run_differential

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
