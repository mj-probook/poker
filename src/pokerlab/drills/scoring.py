"""Decision-ε scoring rule (plan §1 definitions; impl doc §1; Slice E).

The single grading primitive for drills (and, later, tier-1/2 HH grading): read
ONLY a normalized `types.Solution` — never a vendor format (CLAUDE.md hard rule).

    correct  iff  ev_loss <= max(0.005 * pot_bb, 0.1)   (the ε threshold, in bb)
                  OR  frequency(chosen) >= 0.05          (mixed-spot acceptance)

`ev_loss` is the bb the chosen action concedes to the highest-EV action; in a
mixed spot both branches are ~indifferent, so either clears via the ε floor or
the 5% frequency escape hatch.
"""

from __future__ import annotations

from pokerlab.types import Score, Solution

EV_FLOOR_BB = 0.1        # absolute ε floor (bb)
POT_EPS_FRAC = 0.005     # ε as a fraction of pot
MIX_FREQ = 0.05          # frequency at/above which any action is acceptable


def epsilon(pot_bb: float) -> float:
    """The decision-ε for a pot of `pot_bb` big blinds."""
    return max(POT_EPS_FRAC * float(pot_bb), EV_FLOOR_BB)


def score(solution: Solution, chosen: str, pot_bb: float) -> Score:
    """Grade a `chosen` action against `solution` under the decision-ε rule.

    Raises ValueError if `chosen` is not one of the solution's action labels.
    """
    actions = solution.actions
    if chosen not in actions:
        raise ValueError(
            f"action {chosen!r} not in solution actions {sorted(actions)}"
        )
    best_action = max(actions, key=lambda a: actions[a][0])
    best_ev = actions[best_action][0]
    chosen_ev, chosen_freq = actions[chosen]
    ev_loss = best_ev - chosen_ev
    correct = ev_loss <= epsilon(pot_bb) or chosen_freq >= MIX_FREQ
    return Score(
        correct=bool(correct),
        ev_loss_bb=float(ev_loss),
        best_action=best_action,
        chosen_frequency=float(chosen_freq),
    )
