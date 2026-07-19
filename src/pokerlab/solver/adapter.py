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
                weights.append(my_reach[i])
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
