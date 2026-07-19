"""Canonical grading/drill category vocabulary (plan §5.3 training loop).

Drills OWN this vocabulary; the HH pipeline imports it (dependency direction
hh → drills, matching hh's existing use of drills.scoring). One key format is
shared by drill spot_keys, SM-2 leak_keys, and HH gradings.leak_key so the
leak report, the drill generator, and the scheduler all join on the same
categories:

    formation|street|action|depth

  * preflop jam/fold spots carry a stack-depth BUCKET in {5,8,10,15,20}
    (eff_bb snapped via `snap_depth`); formation/action are the spot's canonical
    pair — SB open-jam -> `SBjam|preflop|jam|15`, BB call-vs-jam ->
    `BBcall|preflop|call|10`. The hero's *choice* (jam vs fold) does not change
    the category — a fold in an SB-jam spot is the SAME leak category as a jam,
    so an HH open-fold leak maps onto exactly the drill the generator emits.
  * postflop spots have no depth bucket (`-`) and key off the hero's action:
    `6max:LJ|flop|bet|-`.
"""

from __future__ import annotations

DEPTHS: tuple[int, ...] = (5, 8, 10, 15, 20)
NO_DEPTH = "-"

_JAMFOLD_FORMATION = {"SB": "SBjam", "BB": "BBcall"}
_JAMFOLD_ACTION = {"SB": "jam", "BB": "call"}


def snap_depth(eff_bb: float) -> int:
    """Snap an effective stack (bb) to the nearest push/fold syllabus depth."""
    return min(DEPTHS, key=lambda d: abs(d - float(eff_bb)))


def jamfold_category(position: str, eff_bb: float, *, icm: bool = False) -> str:
    """Canonical category for a ≤20bb push/fold spot (SB jam / BB call)."""
    pos = position.upper()
    formation = _JAMFOLD_FORMATION[pos] + (".icm" if icm else "")
    return f"{formation}|preflop|{_JAMFOLD_ACTION[pos]}|{snap_depth(eff_bb)}"


def postflop_category(formation: str, street: str, action_type: str) -> str:
    """Canonical category for a non-push/fold spot (no depth bucket)."""
    return f"{formation}|{street}|{action_type}|{NO_DEPTH}"


def parse_category(key: str) -> tuple[str, str, str, str]:
    """Split a canonical key into (formation, street, action, depth)."""
    formation, street, action, depth = key.split("|")
    return formation, street, action, depth
