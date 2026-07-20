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


def epsilon(pot_bb: float, *, bb_value: float = 1.0) -> float:
    """The decision-ε for a pot of `pot_bb` big blinds, in the payoff currency.

    `bb_value` is what one big blind is worth in the units the solution's EVs
    are denominated in. It is 1.0 for a chip-EV solution (EVs already in bb)
    and the average chip's value for an ICM solution, whose EVs are $-deltas.

    Round-3 finding [P7']. ICM drills were graded by comparing a $-delta (up to
    109.59 on the bubble fixture) against a 0.1**bb** floor — a unit mismatch,
    so the ε branch could never fire except at exactly ev_loss == 0. The other
    branch could not save it either: the ICM solve is essentially pure (max
    second-action frequency 0.0009 across all 338 ICM drills), so the 5%
    mixed-spot hatch opened 0 times out of 338. Both acceptance paths were
    dead, which made ICM grading exact-argmax with NO indifference tolerance —
    pick the action worth a cent less and you were marked wrong.

    The fix deliberately introduces NO new tolerance constant. The SAME rule
    the plan already justifies is converted into the decision's own currency:

        ε_$ = ε_bb × (payout pool / total chips in play, in bb)

    The factor is the average chip's dollar value, so this is scale-free by
    construction — rescaling the prize ladder or the chip denomination moves ε
    and the EVs together (the [E37] normalize-by-pool lesson). On the bubble
    fixture: pool 1000 over 40bb of chips → 1bb = 25$, so the 0.1bb floor
    becomes 2.5$.

    Calibration check, non-best actions falling inside their own ε: chip-EV
    12.3%, ICM 1.8%. The gap is NOT mis-scaling — the loss distributions
    overlap well in ε-multiples (chip-EV p50 7.4× / p90 24.4×; ICM p50 4.3× /
    p90 29.8×). It is a low-tail effect: chip-EV's p10 is 0.78× because the
    chip charts genuinely mix, while ICM's is 1.83× because its solve is
    near-pure. Sharper solve, fewer near-indifferent spots.
    """
    return max(POT_EPS_FRAC * float(pot_bb), EV_FLOOR_BB) * float(bb_value)


def score(solution: Solution, chosen: str, pot_bb: float, *,
          bb_value: float = 1.0) -> Score:
    """Grade a `chosen` action against `solution` under the decision-ε rule.

    `bb_value` converts ε into the solution's payoff currency — see `epsilon`.
    Note the returned `Score.ev_loss_bb` is in THAT currency too, so an ICM
    grading carries an ICM-$ delta, not bb, and must be labelled accordingly.

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
    correct = ev_loss <= epsilon(pot_bb, bb_value=bb_value) or chosen_freq >= MIX_FREQ
    return Score(
        correct=bool(correct),
        ev_loss_bb=float(ev_loss),
        best_action=best_action,
        chosen_frequency=float(chosen_freq),
    )
