"""Shared contracts for the poker lab (plan §3b; impl doc §1).

FROZEN: changes to this module require team-lead sign-off. Every module
(engine, cfr, charts, solver, drills, hh, rebel, env) codes against these
types; scoring and grading read only `Solution`, never vendor formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Rank = int  # 2..14 (14 = ace)
Suit = int  # 0..3
Card = int  # 0..51, rank-major: card = (rank - 2) * 4 + suit

Action = tuple[str, int]  # ("fold"|"check"|"call"|"bet"|"raise"|"allin", amount_chips)

STREETS = ("preflop", "flop", "turn", "river", "showdown")


@dataclass(frozen=True)
class TournamentContext:
    """Minimal tournament fields any drill/grader reads (plan §3b #1).

    Deliberately excludes blind-clock simulation and multi-table state —
    no milestone consumes them.
    """

    payouts: tuple[int, ...]  # remaining prize ladder, cents, descending
    players_remaining: int
    stacks_all: tuple[int, ...]  # chips, all remaining players (ICM input)
    bb: int  # current big blind, chips
    ante: int = 0


@dataclass
class GameState:
    """Single-table, single-hand state. Engine module owns the transitions:
    legal_actions(), apply(), is_terminal(), payoffs(), replay().
    """

    seats: int
    stacks: list[int]  # chips per seat at hand start
    button: int
    board: list[Card] = field(default_factory=list)
    hole: list[tuple[Card, Card] | None] = field(default_factory=list)
    pot: int = 0
    street: str = "preflop"
    action_history: list[tuple[int, Action]] = field(default_factory=list)
    to_act: int | None = None
    tournament: TournamentContext | None = None


@dataclass(frozen=True)
class SpotKey:
    """Library/grading lookup key (plan §1 definitions)."""

    formation: str  # e.g. "BTNopen_BBcall"
    stack_bucket: int  # effective bb bucketed to one of 10/20/40/100
    board_bucket: str  # f"{iso_class}:{texture}", texture from 8-way taxonomy


@dataclass(frozen=True)
class Solution:
    """Normalized solver output — the ONLY type scoring/grading consume."""

    actions: dict[str, tuple[float, float]]  # action label -> (ev_bb, frequency)
    range_ctx: str  # provenance: source config + hash
    source: str  # "chart" | "cfr" | "subgame_solver" | "manual_seed"


@dataclass(frozen=True)
class Score:
    """Result of the decision-ε rule (plan §1; drills/scoring.py)."""

    correct: bool
    ev_loss_bb: float
    best_action: str
    chosen_frequency: float


TIER_CHART = 1  # preflop / ≤20bb jam-fold: chart engine, exact
TIER_SOLVER = 2  # HU or HU-collapsed postflop: solution library, exact
TIER_BEST_AVAILABLE = 3  # genuine multiway postflop: ev_loss is ALWAYS None
