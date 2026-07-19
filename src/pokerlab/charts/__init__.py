"""Chart engine (M1.5; plan §5.1, §2.2; impl doc §3 Slice C).

In-house push/fold + ICM answer keys for the M2 drills and M4 tier-1 grading.
Everything here is self-generated (chart engine + own solves) — nothing
vendor-derived ever enters an answer key (CLAUDE.md hard rule).

Public surface (consumed by Slice E drills):
    icm_equities(stacks, payouts) -> [$ equity per player]
    jamfold_range(position, depth_bb, ante=0) -> {hand_label: Solution}
"""

from .icm import icm_equities
from .jamfold import jamfold_range, solve_jamfold, solve_jamfold_icm

__all__ = [
    "icm_equities",
    "jamfold_range",
    "solve_jamfold",
    "solve_jamfold_icm",
]
