"""CFR family + exploitability instrument (plan §3b #5, M1).

Public API for dependent slices (C: charts jam/fold; D: M3 subgame solver;
G: M5 ReBeL):

    from pokerlab.cfr import (
        Game, build_tree, uniform_profile,          # protocol + tree cache
        KuhnPoker, LeducPoker,                       # reference games
        CFRSolver, MCCFRSolver,                      # solvers
        best_response, nash_conv, exploitability, on_policy_values,  # instrument
    )

A `Game` exposes initial_states / current_player / chance_outcomes /
legal_actions / infoset_key / apply / is_terminal / returns. Implement it and
everything else (solve + score) works unchanged. A `profile` is
`dict[infoset_key -> {action -> probability}]`.
"""

from .cfr import CFRSolver
from .exploit import (
    BestResponse,
    best_response,
    exploitability,
    nash_conv,
    on_policy_values,
)
from .game import (
    CHANCE,
    TERMINAL,
    Game,
    Profile,
    Tree,
    TreeNode,
    build_tree,
    uniform_profile,
)
from .kuhn import KuhnPoker
from .leduc import LeducPoker
from .mccfr import MCCFRSolver

__all__ = [
    "CHANCE",
    "TERMINAL",
    "Game",
    "Profile",
    "Tree",
    "TreeNode",
    "build_tree",
    "uniform_profile",
    "KuhnPoker",
    "LeducPoker",
    "CFRSolver",
    "MCCFRSolver",
    "BestResponse",
    "best_response",
    "nash_conv",
    "exploitability",
    "on_policy_values",
]
