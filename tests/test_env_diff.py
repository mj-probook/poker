"""Slice H exit: SingleEnvAdapter passes the reused M0 PokerKit differential.

Drives seeded random hands through the discrete env encoding (decode/mask) and
compares to PokerKit via the exact same oracle + odd-chip classifier as Slice A.
Proves the vectorized-env action path reproduces engine mechanics.
"""

import pytest

from pokerlab.env.vec import single_env_action
from tests.diff_harness import run_differential

_QUIRK_CEILING = 0.01


def test_single_env_adapter_matches_pokerkit_10k() -> None:
    exact, oddchip, mismatches = run_differential(10_000, action_fn=single_env_action)
    total = exact + oddchip + len(mismatches)
    assert total == 10_000
    assert not mismatches, (
        f"{len(mismatches)} env-path disagreements with PokerKit:\n"
        + "\n".join(mismatches[:5])
    )
    assert oddchip <= total * _QUIRK_CEILING


@pytest.mark.slow
def test_single_env_adapter_matches_pokerkit_200k() -> None:
    exact, oddchip, mismatches = run_differential(200_000, action_fn=single_env_action)
    total = exact + oddchip + len(mismatches)
    assert total == 200_000
    assert not mismatches
    assert oddchip <= total * _QUIRK_CEILING
