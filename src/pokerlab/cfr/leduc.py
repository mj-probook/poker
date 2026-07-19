"""Leduc hold'em (plan §7 L1; impl doc §3 Slice B).

6-card deck: indices 0..5, rank = card // 2 (0=J, 1=Q, 2=K), two of each rank.
Two players ante 1. Round 1 betting (bet size 2), then one public card, round 2
betting (bet size 4). Max 2 raises per round. Showdown: pairing the public rank
wins; else higher private rank; equal ranks split.

Encoding matches OpenSpiel `leduc_poker` exactly (actions 0=Fold, 1=Call,
2=Raise; deal P0 then P1 then public as absolute card indices; chip units) so
the openspiel cross-check replays histories one-for-one. Verified against
OpenSpiel's 5520 terminals / 936 infosets and its ±13 return range.
"""

from __future__ import annotations

from dataclasses import dataclass

from .game import CHANCE, TERMINAL, Game

FOLD, CALL, RAISE = 0, 1, 2
_DECK = tuple(range(6))
_RAISE_SIZE = {1: 2, 2: 4}
_MAX_RAISES = 2


@dataclass(frozen=True)
class LeducState:
    private: tuple[int, ...] = ()     # private[0] -> P0, private[1] -> P1
    public: int | None = None
    rnd: int = 1                      # current betting round (1 or 2)
    contrib: tuple[int, int] = (1, 1)  # chips in per player (antes posted)
    to_move: int = 0
    raises: int = 0                   # raises made this round
    acted: int = 0                    # actions since round start / last raise
    bets: tuple[int, ...] = ()        # full public betting history (for infoset id)
    folder: int | None = None
    done: bool = False


class LeducPoker(Game):
    num_players = 2

    def initial_states(self) -> list[tuple[LeducState, float]]:
        return [(LeducState(), 1.0)]

    # -- node typing --------------------------------------------------------- #
    def _dealing_private(self, s: LeducState) -> bool:
        return len(s.private) < 2

    def _dealing_public(self, s: LeducState) -> bool:
        return s.rnd == 2 and s.public is None and not s.done

    def current_player(self, s: LeducState) -> int:
        if s.done:
            return TERMINAL
        if self._dealing_private(s) or self._dealing_public(s):
            return CHANCE
        return s.to_move

    def is_terminal(self, s: LeducState) -> bool:
        return s.done

    def chance_outcomes(self, s: LeducState) -> list[tuple[int, float]]:
        used = set(s.private)
        if s.public is not None:
            used.add(s.public)
        remaining = [c for c in _DECK if c not in used]
        p = 1.0 / len(remaining)
        return [(c, p) for c in remaining]

    # -- betting ------------------------------------------------------------- #
    def _facing_bet(self, s: LeducState) -> bool:
        return s.contrib[0] != s.contrib[1]

    def legal_actions(self, s: LeducState) -> list[int]:
        actions = []
        if self._facing_bet(s):
            actions.append(FOLD)
        actions.append(CALL)
        if s.raises < _MAX_RAISES:
            actions.append(RAISE)
        return actions

    def infoset_key(self, s: LeducState, player: int) -> str:
        # OpenSpiel's infoset partition exactly: own private card + public card
        # + full public betting history. `bets` never merges distinct histories.
        return f"P{s.private[player]}|B{s.public}|{s.bets}"

    def apply(self, s: LeducState, action: int) -> LeducState:
        # chance: deal private cards then the public card
        if self._dealing_private(s):
            return LeducState(private=s.private + (action,), public=s.public,
                              rnd=s.rnd, contrib=s.contrib, to_move=0,
                              bets=s.bets)
        if self._dealing_public(s):
            return LeducState(private=s.private, public=action, rnd=2,
                              contrib=s.contrib, to_move=0, raises=0, acted=0,
                              bets=s.bets)

        cur, opp = s.to_move, 1 - s.to_move
        c = list(s.contrib)
        bets = s.bets + (action,)
        if action == FOLD:
            return LeducState(private=s.private, public=s.public, rnd=s.rnd,
                              contrib=s.contrib, to_move=opp, raises=s.raises,
                              acted=s.acted, bets=bets, folder=cur, done=True)
        if action == RAISE:
            c[cur] = c[opp] + _RAISE_SIZE[s.rnd]
            return LeducState(private=s.private, public=s.public, rnd=s.rnd,
                              contrib=(c[0], c[1]), to_move=opp,
                              raises=s.raises + 1, acted=1, bets=bets)
        # CALL (match; may be a check when contributions already equal)
        c[cur] = c[opp]
        acted = s.acted + 1
        if acted >= 2:  # round closes
            if s.rnd == 1:
                # advance to public deal (round 2); public dealt by chance next
                return LeducState(private=s.private, public=None, rnd=2,
                                  contrib=(c[0], c[1]), to_move=0,
                                  raises=0, acted=0, bets=bets)
            return LeducState(private=s.private, public=s.public, rnd=2,
                              contrib=(c[0], c[1]), to_move=opp, bets=bets,
                              done=True)
        return LeducState(private=s.private, public=s.public, rnd=s.rnd,
                          contrib=(c[0], c[1]), to_move=opp,
                          raises=s.raises, acted=acted, bets=bets)

    # -- payoffs ------------------------------------------------------------- #
    def _winner(self, s: LeducState) -> int | None:
        """Return 0 or 1, or None for a split."""
        r0, r1, rp = s.private[0] // 2, s.private[1] // 2, s.public // 2
        p0, p1 = (r0 == rp), (r1 == rp)
        if p0 != p1:
            return 0 if p0 else 1
        if r0 != r1:
            return 0 if r0 > r1 else 1
        return None

    def returns(self, s: LeducState) -> list[float]:
        if s.folder is not None:
            w = 1 - s.folder
            amt = float(s.contrib[s.folder])
            return [amt, -amt] if w == 0 else [-amt, amt]
        w = self._winner(s)
        if w is None:
            return [0.0, 0.0]
        amt = float(s.contrib[1 - w])
        return [amt, -amt] if w == 0 else [-amt, amt]
