"""Adapt a solved subgame into the frozen `types.Solution` (impl doc §1, §3 D).

`Solution` is the ONLY type scoring/grading read (types.py). This module turns a
solved `SubgameSolver` into the **root strategy per hand-class**: for each 169
preflop class the acting player holds, a `Solution` giving every root action's
``(ev_bb, frequency)``, range-weighted across the class' concrete combos.

Frequencies come straight from the average strategy; EVs are real bb values
(counterfactual value ÷ opponent reach). ``source="subgame_solver"`` and
``range_ctx`` carries provenance (spot key + solver version + measured
exploitability) so a grader can trace exactly which solve answered it.
"""

from __future__ import annotations

import numpy as np

from pokerlab.charts import hands
from pokerlab.solver.subgame import COMBO_INDEX, SubgameSolver
from pokerlab.types import Solution


def root_solution_by_class(
    solver: SubgameSolver, player: int = 0, range_ctx: str = ""
) -> dict[str, Solution]:
    """{hand_class -> Solution} for the root actor's range (``player``)."""
    labels, evs, freqs, my_reach = solver.root_action_evs(player)
    valid_opp = solver.valid_opponent_reach(player)
    live_pos = {full: i for i, full in enumerate(solver.live.tolist())}
    out: dict[str, Solution] = {}
    for cls in hands.HAND_CLASSES:
        idxs, weights = [], []
        for a, b in hands.card_combos(cls):
            i = live_pos.get(COMBO_INDEX[(min(a, b), max(a, b))])
            # A combo that blocks the opponent's ENTIRE range has no legal
            # matchup: its EVs are all 0/0 and its average strategy falls back
            # to uniform. Emitting that as a Solution manufactures an answer key
            # where every action scores correct with zero EV loss, so such
            # combos are excluded here at the source (wave-2 [E26]).
            if i is not None and my_reach[i] > 0 and valid_opp[i] > 0:
                idxs.append(i)
                # Weight by P(hold this combo) x P(a legal matchup exists), not
                # by reach alone. A combo the opponent's range blocks heavily
                # contributes to the class average as strongly as an unblocked
                # one under reach-only weighting, even though it reaches a
                # showdown far less often (wave-3 [M8]).
                #
                # The per-combo EVs are ALREADY conditional — subgame.py divides
                # by valid_opp before returning them — so this belongs on the
                # weight and NOWHERE else; applying it twice would divide the
                # blocking out of the number entirely.
                #
                # Under UNIFORM ranges this changes nothing, and provably so:
                # valid_opp is then identical for every live combo (each two-card
                # hand blocks the same count), so it scales all of a class'
                # weights by one constant and cancels in the normalized mean.
                # Measured within-class spread is exactly 0.000000 uniform, and
                # 0.028 mean / 0.057 max once the opponent range is non-uniform.
                # So this is correct now and load-bearing the moment tier-2 stops
                # assuming uniform ranges (see [M4]).
                weights.append(my_reach[i] * valid_opp[i])
        if not idxs:
            continue  # class absent from this player's range, or fully blocked
        w = np.array(weights)
        wsum = w.sum()
        actions: dict[str, tuple[float, float]] = {}
        for a, label in enumerate(labels):
            freq = float((w * freqs[a, idxs]).sum() / wsum)
            ev = float((w * evs[a][idxs]).sum() / wsum)
            actions[label] = (ev, freq)
        out[cls] = Solution(actions=actions, range_ctx=range_ctx, source="subgame_solver")
    return out
