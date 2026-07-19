"""Parsed hand-history model shared by the PokerStars and GGPoker parsers (Slice F, impl doc §3).

A `ParsedHand` is the bridge between raw HH text and the Slice-A engine: it
carries a ready-to-replay `HandSetup` + engine action list (including the
per-player-`ante` vs `bb_ante` structure classification — see `hh._common`),
plus the results *stated in the HH text* (per-seat contributions, collected
amounts, uncalled returns, total pot) so a replay can be validated against
ground truth.

Seat convention: HH seats are compacted in ascending seat-number order onto
engine indices ``0..n-1`` (ascending-with-wraparound == clockwise on both
sites), so the engine's derived positions (SB = ``(button+1)%n`` for n>=3)
line up with the physical table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pokerlab.engine.state import HandSetup
from pokerlab.types import Action


@dataclass
class ParsedHand:
    site: str
    setup: HandSetup
    actions: list[tuple[int, Action]]
    seat_names: tuple[str, ...]  # engine index -> anonymized name
    hero: int  # engine index of the hero (the "Dealt to" player)
    contributed: tuple[int, ...]  # net chips each seat put in the pot (post-uncalled)
    collected: tuple[int, ...]  # chips each seat collected from pot(s)
    uncalled: int  # total chips returned as uncalled bets
    total_pot: int  # contested pot as stated in SUMMARY
    hand_id: str = ""
    tournament_id: str = ""
    level: str = ""
    meta: dict = field(default_factory=dict)

    def stated_final_stacks(self) -> tuple[int, ...]:
        """Final stack per seat implied by the HH text (independent of replay)."""
        start = self.setup.stacks
        return tuple(
            start[i] - self.contributed[i] + self.collected[i]
            for i in range(len(start))
        )
