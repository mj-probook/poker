"""Finite imperfect-information game protocol + generic tree cache (plan §3b #5).

The CFR family and the exploitability/best-response utility (the instrument
M3/M5/M6/M7 exits depend on) are written against `Game`. Kuhn and Leduc
implement it here; M3 subgames adapt to it later.

A *state* is opaque to this module — each game owns its own immutable state
type. The protocol exposes, for any state:

  initial_states() -> [(state, prob)]   roots with prior weights (sum to 1).
                                        Full games return a single chance root
                                        [(root, 1.0)]; M3 subgames return a
                                        belief distribution over post-deal states.
  current_player(state) -> int          0..n-1, or CHANCE / TERMINAL sentinels.
  chance_outcomes(state) -> [(action, prob)]   at chance nodes only.
  legal_actions(state) -> [action]      at decision nodes.
  infoset_key(state, player) -> str     the acting player's information set.
  apply(state, action) -> state         transition (pure; never mutates input).
  is_terminal(state) -> bool
  returns(state) -> [payoff_per_player]  at terminal nodes (chip deltas).

`build_tree(game)` walks the protocol once into a flat, arithmetic-only cache
that the solver and exploit utilities iterate cheaply (no game logic per pass).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# Node ownership sentinels (OpenSpiel convention: chance = -1, terminal = -4).
CHANCE = -1
TERMINAL = -4

State = Any
Action = Any
# A behavioural strategy profile: infoset key -> {action -> probability}.
Profile = dict[str, dict[Action, float]]


class Game(ABC):
    """Base class for finite imperfect-info games consumed by cfr/exploit."""

    num_players: int = 2

    @abstractmethod
    def initial_states(self) -> list[tuple[State, float]]:
        """Root states paired with their prior probabilities (sum to 1)."""

    @abstractmethod
    def current_player(self, state: State) -> int:
        """Acting player (0..n-1), or CHANCE / TERMINAL."""

    @abstractmethod
    def chance_outcomes(self, state: State) -> list[tuple[Action, float]]:
        """(outcome action, probability) pairs at a chance node."""

    @abstractmethod
    def legal_actions(self, state: State) -> list[Action]:
        """Legal actions at a decision node, in a stable canonical order."""

    @abstractmethod
    def infoset_key(self, state: State, player: int) -> str:
        """Information-set key from `player`'s perspective (private + public)."""

    @abstractmethod
    def apply(self, state: State, action: Action) -> State:
        """Return the successor state; never mutate `state`."""

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        ...

    @abstractmethod
    def returns(self, state: State) -> list[float]:
        """Per-player payoffs (chip deltas) at a terminal state."""


# --------------------------------------------------------------------------- #
# Flat tree cache — built once, iterated many times by cfr.py / exploit.py.
# --------------------------------------------------------------------------- #
@dataclass
class TreeNode:
    kind: str                       # "terminal" | "chance" | "decision"
    player: int                     # decision: acting player; else CHANCE/TERMINAL
    infoset: str | None             # decision only
    actions: list                   # decision: legal actions; chance: outcome actions
    children: list[int]             # node indices aligned with `actions`
    chance_probs: list[float] | None
    payoff: tuple[float, ...] | None


@dataclass
class Tree:
    num_players: int
    nodes: list[TreeNode]
    roots: list[tuple[int, float]]              # (node index, prior prob)
    infoset_actions: dict[str, list]            # infoset -> canonical action order
    infoset_nodes: dict[str, list[int]]         # infoset -> node indices sharing it
    infoset_player: dict[str, int] = field(default_factory=dict)

    @property
    def infosets(self) -> list[str]:
        return list(self.infoset_actions.keys())


def build_tree(game: Game) -> Tree:
    """Expand `game` into a flat node list with shared-infoset bookkeeping."""
    nodes: list[TreeNode] = []
    infoset_actions: dict[str, list] = {}
    infoset_nodes: dict[str, list[int]] = {}
    infoset_player: dict[str, int] = {}

    def build(state: State) -> int:
        idx = len(nodes)
        if game.is_terminal(state):
            nodes.append(TreeNode("terminal", TERMINAL, None, [], [], None,
                                  tuple(game.returns(state))))
            return idx
        player = game.current_player(state)
        if player == CHANCE:
            outcomes = game.chance_outcomes(state)
            actions = [a for a, _ in outcomes]
            probs = [p for _, p in outcomes]
            node = TreeNode("chance", CHANCE, None, actions, [], probs, None)
            nodes.append(node)
            node.children = [build(game.apply(state, a)) for a in actions]
            return idx
        # decision node
        actions = list(game.legal_actions(state))
        key = game.infoset_key(state, player)
        if key not in infoset_actions:
            infoset_actions[key] = actions
            infoset_nodes[key] = []
            infoset_player[key] = player
        elif infoset_actions[key] != actions:
            raise ValueError(
                f"infoset {key!r} has inconsistent legal actions "
                f"{infoset_actions[key]} vs {actions}")
        node = TreeNode("decision", player, key, actions, [], None, None)
        nodes.append(node)
        infoset_nodes[key].append(idx)
        node.children = [build(game.apply(state, a)) for a in actions]
        return idx

    roots = [(build(s), p) for s, p in game.initial_states()]
    return Tree(game.num_players, nodes, roots,
                infoset_actions, infoset_nodes, infoset_player)


def uniform_profile(tree: Tree) -> Profile:
    """Uniform behavioural strategy over each infoset's legal actions."""
    prof: Profile = {}
    for key, actions in tree.infoset_actions.items():
        p = 1.0 / len(actions)
        prof[key] = {a: p for a in actions}
    return prof
