"""Kuhn poker (plan §7 L1; impl doc §3 Slice B).

3 cards J<Q<K encoded 0<1<2, one dealt to each player, one unused. Both ante 1.
P0 acts first. Action 0 = Pass (check/fold), 1 = Bet (bet/call) — matching the
OpenSpiel `kuhn_poker` encoding so the openspiel cross-check maps cleanly.

Terminal chip deltas (hand-derived, cross-checked vs OpenSpiel terminals):
  pass,pass          -> showdown for 1   (higher card +1)
  bet,fold           -> +1 to P0         (P1 folds)
  bet,call           -> showdown for 2
  pass,bet,fold      -> -1 to P0         (P0 folds)
  pass,bet,call      -> showdown for 2
"""

from __future__ import annotations

from dataclasses import dataclass

from .game import CHANCE, TERMINAL, Game

PASS, BET = 0, 1
_TERMINAL_HISTORIES = {(0, 0), (1, 0), (1, 1), (0, 1, 0), (0, 1, 1)}


@dataclass(frozen=True)
class KuhnState:
    cards: tuple[int, ...] = ()      # cards[0] -> P0, cards[1] -> P1
    history: tuple[int, ...] = ()    # player actions after the deal


class KuhnPoker(Game):
    num_players = 2

    def initial_states(self) -> list[tuple[KuhnState, float]]:
        return [(KuhnState(), 1.0)]

    def current_player(self, state: KuhnState) -> int:
        if len(state.cards) < 2:
            return CHANCE
        if self.is_terminal(state):
            return TERMINAL
        return len(state.history) % 2

    def chance_outcomes(self, state: KuhnState) -> list[tuple[int, float]]:
        dealt = set(state.cards)
        remaining = [c for c in (0, 1, 2) if c not in dealt]
        p = 1.0 / len(remaining)
        return [(c, p) for c in remaining]

    def legal_actions(self, state: KuhnState) -> list[int]:
        return [PASS, BET]

    def infoset_key(self, state: KuhnState, player: int) -> str:
        hist = "".join("p" if a == PASS else "b" for a in state.history)
        return f"{state.cards[player]}{hist}"

    def apply(self, state: KuhnState, action: int) -> KuhnState:
        if len(state.cards) < 2:
            return KuhnState(cards=state.cards + (action,), history=state.history)
        return KuhnState(cards=state.cards, history=state.history + (action,))

    def is_terminal(self, state: KuhnState) -> bool:
        return len(state.cards) == 2 and state.history in _TERMINAL_HISTORIES

    def returns(self, state: KuhnState) -> list[float]:
        h = state.history
        if h == (1, 0):          # bet, fold
            return [1.0, -1.0]
        if h == (0, 1, 0):       # pass, bet, fold
            return [-1.0, 1.0]
        amount = 2.0 if h in ((1, 1), (0, 1, 1)) else 1.0
        r0 = amount if state.cards[0] > state.cards[1] else -amount
        return [r0, -r0]
