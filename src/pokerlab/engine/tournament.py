"""Tournament helpers over the frozen `TournamentContext` (plan §3b; impl §3 A).

Deliberately minimal — no blind-clock simulation or multi-table state (no
milestone consumes them). Just the read-only quantities drills/graders need:
effective stack in big blinds, and validation of a monotone blind/ante schedule.
"""

from __future__ import annotations

from typing import NamedTuple

from pokerlab.types import TournamentContext


def effective_bb(stack: int, tc: TournamentContext) -> float:
    """A stack expressed in big blinds (the drill/ICM working unit)."""
    if tc.bb <= 0:
        raise ValueError("big blind must be positive")
    return stack / tc.bb


def effective_bb_all(tc: TournamentContext) -> tuple[float, ...]:
    """Every remaining player's stack in big blinds."""
    return tuple(effective_bb(s, tc) for s in tc.stacks_all)


def m_ratio(stack: int, tc: TournamentContext, seats: int) -> float:
    """Harrington M: orbits survivable = stack / (sb + bb + seats*ante).

    ``seats`` is players dealt in per orbit (antes are per-player here).
    """
    cost = tc.bb + tc.bb // 2 + seats * tc.ante
    if cost <= 0:
        raise ValueError("orbit cost must be positive")
    return stack / cost


class BlindLevel(NamedTuple):
    """One level of a tournament blind schedule (chips)."""

    sb: int
    bb: int
    ante: int = 0


def is_monotone_schedule(levels: list[BlindLevel]) -> bool:
    """True iff sb, bb and ante are each non-decreasing across the schedule
    and every level is internally sane (0 < sb <= bb, ante >= 0)."""
    for lvl in levels:
        if not (0 < lvl.sb <= lvl.bb and lvl.ante >= 0):
            return False
    for a, b in zip(levels, levels[1:]):
        if b.sb < a.sb or b.bb < a.bb or b.ante < a.ante:
            return False
    return True
