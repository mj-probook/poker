"""Depth-limited Leduc with oracle leaf values (plan §7 L3; impl doc §3 Slice G).

The ReBeL mechanism's skeleton at verifiable scale. A *depth-limited* solve keeps
only the round-1 trunk of Leduc and replaces the round-2 continuation with a
**leaf value** — here read off a full-game CFR+ solution σ* (the oracle). This is
the sanity harness that de-risks everything before a neural net stands in for the
oracle.

`DepthLimitedLeducOracle` is plain `LeducPoker` truncated at the round-1/round-2
boundary: the public-card deal node becomes a pseudo-terminal whose payoff is the
value of continuing under σ* with the concrete private cards. Because it is a
`Game`, the existing `build_tree` / `CFRSolver` / `nash_conv` machinery solves and
scores it unchanged — the trunk is just a small game.

Correctness comes from the leaf value: for concrete cards (c0, c1) at a round-2
entry, `continuation_value` evaluates the exact expectation of σ*'s round-2 play
over the public-card deal. Trunk CFR against these leaves converges to the full
game's round-1 equilibrium (see tests/test_rebel_depth_limited.py).
"""

from __future__ import annotations

from pokerlab.cfr.game import CHANCE, Profile, Game
from pokerlab.cfr.leduc import LeducPoker, LeducState


def round2_entry_state(contrib: tuple[int, int], bets: tuple[int, ...],
                       c0: int, c1: int) -> LeducState:
    """The state the instant round-1 betting closes: round 2, public not yet dealt."""
    return LeducState(private=(c0, c1), public=None, rnd=2, contrib=contrib,
                      to_move=0, raises=0, acted=0, bets=bets)


def continuation_value(game: LeducPoker, sigma: Profile, state: LeducState,
                       player: int = 0) -> float:
    """Expected payoff to `player` from `state`, playing `sigma` to the showdown.

    Chance nodes (the public-card deal) are averaged; decision nodes follow the
    fixed profile. Used to price the depth limit's leaves under the oracle.
    """
    if game.is_terminal(state):
        return game.returns(state)[player]
    cp = game.current_player(state)
    if cp == CHANCE:
        return sum(p * continuation_value(game, sigma, game.apply(state, a), player)
                   for a, p in game.chance_outcomes(state))
    dist = sigma[game.infoset_key(state, cp)]
    return sum(dist[a] * continuation_value(game, sigma, game.apply(state, a), player)
               for a in game.legal_actions(state))


def _is_round2_entry(s: LeducState) -> bool:
    """True at the public-card deal node that opens round 2 — the depth limit."""
    return s.rnd == 2 and s.public is None and not s.done


class DepthLimitedLeducOracle(LeducPoker):
    """Leduc truncated at round-2 entry; leaf payoff = σ* continuation value."""

    def __init__(self, sigma: Profile):
        self.sigma = sigma
        self._base = LeducPoker()   # untruncated dynamics, for continuation values
        self._cache: dict[tuple, float] = {}

    def current_player(self, s: LeducState) -> int:
        if _is_round2_entry(s):
            from pokerlab.cfr.game import TERMINAL
            return TERMINAL
        return super().current_player(s)

    def is_terminal(self, s: LeducState) -> bool:
        return _is_round2_entry(s) or super().is_terminal(s)

    def returns(self, s: LeducState) -> list[float]:
        if not _is_round2_entry(s):
            return super().returns(s)
        key = (s.bets, s.private, s.contrib)
        v0 = self._cache.get(key)
        if v0 is None:
            v0 = continuation_value(self._base, self.sigma, s, player=0)
            self._cache[key] = v0
        return [v0, -v0]
