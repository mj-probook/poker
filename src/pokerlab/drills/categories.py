"""Canonical grading/drill category vocabulary (plan §5.3 training loop).

Drills OWN this vocabulary; the HH pipeline imports it (dependency direction
hh → drills, matching hh's existing use of drills.scoring). One key format is
shared by drill leak_keys, SM-2 leak_keys, and HH gradings.leak_key so the
leak report, the drill generator, and the scheduler all join on the same
categories:

    formation|street|action|depth

  * preflop jam/fold spots carry a stack-depth BUCKET in {5,8,10,15,20}
    (eff_bb snapped via `snap_depth`); formation/action are the spot's canonical
    pair — SB open-jam -> `SBjam|preflop|jam|15`, BB call-vs-jam ->
    `BBcall|preflop|call|10`. The hero's *choice* (jam vs fold) does not change
    the category — a fold in an SB-jam spot is the SAME leak category as a jam,
    so an HH open-fold leak maps onto exactly the drill the generator emits.
  * jam/fold spots ALSO carry an ante bucket in {0, 0.125, 0.25} bb/player
    (`snap_ante`), suffixed onto the depth token as `10a0.125`. See the
    taxonomy note below.
  * postflop spots have no depth bucket (`-`) and key off the hero's action:
    `6max:LJ|flop|bet|-`.
"""

from __future__ import annotations

DEPTHS: tuple[int, ...] = (5, 8, 10, 15, 20)
NO_DEPTH = "-"

# Ante buckets, bb/player (round-3 findings [E49]/[A5]). The generator solved
# jam/fold ante-FREE while the grader passed the hand's real ante and the key
# carried no ante component at all, so grader and drill disagreed for
# essentially every real MTT hand — 15 chart flips at 0.125bb/player, 25 at
# 0.25. That is the same invariant break [E33] fixed on the depth axis (snap
# ONCE, key and answer key off the same snapped value), on a second axis.
#
# Two honesty notes on this bucket set, recorded verbatim because both cut
# against the tidy version of the story:
#
#   1. The residual is 3 ε-misgrades, not 0. Snapping is an approximation, and
#      three (position, depth, ante) cells still grade a hand against a chart
#      solved at a neighbouring ante. Three is not zero and this taxonomy does
#      not claim otherwise.
#   2. Boundary PLACEMENT dominates bucket COUNT — a 4-bucket design measured
#      WORSE. Over a realistic ante axis (BB antes of 1–1.5bb across 7–9 payers,
#      plus the no-ante and 2bb tails), total ε-misgrades were: 2-bucket
#      {0,.125} = 31, 3-bucket {0,.125,.25} = 3, 4-bucket {0,.083,.167,.25} = 5
#      (/tmp/x_a5.py). Adding a bucket is not automatically an improvement; do
#      not "refine" this set without re-running that measurement.
#
# ante=0 deliberately produces the byte-identical pre-rev key (no suffix), so
# the rev does not orphan existing sr_state / gradings rows.
ANTES: tuple[float, ...] = (0.0, 0.125, 0.25)

_JAMFOLD_FORMATION = {"SB": "SBjam", "BB": "BBcall"}
_JAMFOLD_ACTION = {"SB": "jam", "BB": "call"}


def snap_depth(eff_bb: float) -> int:
    """Snap an effective stack (bb) to the nearest push/fold syllabus depth."""
    return min(DEPTHS, key=lambda d: abs(d - float(eff_bb)))


def snap_ante(ante_bb: float) -> float:
    """Snap a per-player ante (bb) to the nearest syllabus ante bucket.

    THE shared snap for the ante axis: the grader and the drill generator must
    both call this, or they solve different charts for the same key — the
    [E49] break this exists to close.
    """
    return min(ANTES, key=lambda a: abs(a - float(ante_bb)))


def _depth_token(eff_bb: float, ante_bb: float) -> str:
    """The key's depth field: bucketed depth, ante bucket appended when nonzero."""
    ante = snap_ante(ante_bb)
    return f"{snap_depth(eff_bb)}" + (f"a{ante:g}" if ante else "")


def jamfold_category(position: str, eff_bb: float, *, icm: bool = False,
                     ante_bb: float = 0.0) -> str:
    """Canonical category for a ≤20bb push/fold spot (SB jam / BB call)."""
    pos = position.upper()
    formation = _JAMFOLD_FORMATION[pos] + (".icm" if icm else "")
    token = _depth_token(eff_bb, ante_bb)
    return f"{formation}|preflop|{_JAMFOLD_ACTION[pos]}|{token}"


def postflop_category(formation: str, street: str, action_type: str) -> str:
    """Canonical category for a non-push/fold spot (no depth bucket)."""
    return f"{formation}|{street}|{action_type}|{NO_DEPTH}"


def parse_category(key: str) -> tuple[str, str, str, str]:
    """Split a canonical key into (formation, street, action, depth)."""
    formation, street, action, depth = key.split("|")
    return formation, street, action, depth
