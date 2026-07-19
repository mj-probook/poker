"""Slice A: TournamentContext helpers + blind/ante progression properties."""

import dataclasses

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pokerlab.engine.tournament import (
    BlindLevel,
    effective_bb,
    effective_bb_all,
    is_monotone_schedule,
    m_ratio,
)
from pokerlab.types import TournamentContext


def _tc(bb=200, ante=25, stacks=(4000, 3000, 2000)):
    return TournamentContext(payouts=(500000, 300000, 200000), players_remaining=len(stacks),
                             stacks_all=tuple(stacks), bb=bb, ante=ante)


def test_effective_bb_and_all() -> None:
    tc = _tc(bb=200, stacks=(4000, 3000, 2000))
    assert effective_bb(4000, tc) == 20.0
    assert effective_bb_all(tc) == (20.0, 15.0, 10.0)


def test_effective_bb_rejects_bad_bb() -> None:
    with pytest.raises(ValueError):
        effective_bb(1000, _tc(bb=0))


def test_m_ratio() -> None:
    # bb=200 -> sb=100, ante 25, 9-handed: cost = 200+100+9*25 = 525
    tc = _tc(bb=200, ante=25)
    assert m_ratio(5250, tc, seats=9) == pytest.approx(10.0)


def test_payout_vector_is_immutable() -> None:
    tc = _tc()
    assert isinstance(tc.payouts, tuple)  # can't be mutated in place
    with pytest.raises(dataclasses.FrozenInstanceError):
        tc.bb = 400  # type: ignore[misc]
    with pytest.raises(TypeError):
        tc.payouts[0] = 1  # type: ignore[index]  # tuples reject item assignment


def test_monotone_schedule_accepts_and_rejects() -> None:
    good = [BlindLevel(50, 100, 0), BlindLevel(100, 200, 25), BlindLevel(150, 300, 50)]
    assert is_monotone_schedule(good)
    # blinds go down at level 3
    bad = [BlindLevel(100, 200, 25), BlindLevel(75, 150, 25)]
    assert not is_monotone_schedule(bad)
    # sb > bb is internally invalid
    assert not is_monotone_schedule([BlindLevel(300, 200, 0)])


@given(
    base_bb=st.integers(min_value=2, max_value=500),
    steps=st.lists(st.integers(min_value=0, max_value=400), min_size=1, max_size=12),
    ante_steps=st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=12),
    stack=st.integers(min_value=1, max_value=10_000_000),
)
def test_rising_schedule_is_monotone_and_shrinks_effective_stack(
    base_bb, steps, ante_steps, stack
) -> None:
    # Build a schedule with non-decreasing bb (sb = bb//2, sb>=1) and ante.
    bb, ante = max(2, base_bb), 0
    levels, effs = [], []
    for db, da in zip(steps, ante_steps + [0] * len(steps)):
        bb += db
        ante += da
        lvl = BlindLevel(sb=max(1, bb // 2), bb=bb, ante=ante)
        levels.append(lvl)
        effs.append(stack / lvl.bb)
    assert is_monotone_schedule(levels)
    # a fixed stack is worth non-increasing big blinds as blinds rise
    assert all(a >= b - 1e-9 for a, b in zip(effs, effs[1:]))
