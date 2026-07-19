"""Safe round-2 re-solving via the CFR-D gadget (plan §7 L3; impl doc §3 Slice G).

Naively re-solving a subgame at a belief is UNSAFE: the round-2 game has many
equilibria of equal value, and the one CFR happens to pick is generally
inconsistent with the trunk's incentives, so a best-responder exploits it by
deviating earlier to reach the subgame with an off-blueprint range. (Feeding even
σ*'s own round-1 through a naive re-solve leaves NashConv ≈ 0.2.)

The fix (Burch–Johanson–Bowling 2014, and the mechanism DeepStack/ReBeL use):
re-solve inside a *gadget* that preserves the opponent's counterfactual values.
The opponent, per private hand h, chooses to TERMINATE — taking a fixed payoff
equal to the counterfactual value the trunk promised them, cfv_opp[h] — or to
FOLLOW into the subgame. In equilibrium the resolver must hold the opponent to
that value for every hand, which pins down a resolver strategy that stays
non-exploitable in the full game.

Normalization matches the trunk's cfv convention (cfv_p[h] = Σ_{h'} DEAL·r_opp[h']
· U_p): in the FOLLOW branch chance deals the resolver's hand h_R with the
UNnormalized weight DEAL[h,h_R]·reachR[h_R] and sends the remaining mass to a
zero-payoff dummy, so the FOLLOW value equals cfv_opp[h] in the same units as the
TERMINATE payoff. We run the gadget once per resolver (0 and 1) and keep each
resolver's own round-2 strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pokerlab.cfr.cfr import CFRSolver
from pokerlab.cfr.game import CHANCE, TERMINAL, Game, Profile, build_tree
from pokerlab.cfr.leduc import LeducPoker, LeducState

from .depth_limited import round2_entry_state
from .pbs import DEAL, NUM_CARDS

_BASE = LeducPoker()


@dataclass(frozen=True)
class _GRoot:
    h: int          # opponent's private hand at the gadget root


@dataclass(frozen=True)
class _GDeal:
    h: int          # opponent hand fixed; chance now deals the resolver's hand


@dataclass(frozen=True)
class _GTerm:
    h: int          # opponent terminated with hand h


@dataclass(frozen=True)
class _GDummy:
    pass


_TERM, _FOLLOW, _DUMMY = "T", "F", "dummy"


class SafeResolveGadget(Game):
    """Round-2 re-solve preserving the opponent's counterfactual values."""

    num_players = 2

    def __init__(self, contrib, bets, reachR, cfv_opp, resolver: int):
        self.contrib = tuple(contrib)
        self.bets = tuple(bets)
        self.resolver = resolver
        self.opp = 1 - resolver
        self.reachR = np.asarray(reachR, dtype=float)
        self.cfv_opp = np.asarray(cfv_opp, dtype=float)
        self._roots = [(_GRoot(h), 1.0 / NUM_CARDS) for h in range(NUM_CARDS)]

    def initial_states(self):
        return self._roots

    def current_player(self, s):
        if isinstance(s, _GRoot):
            return self.opp                 # opponent chooses TERMINATE / FOLLOW
        if isinstance(s, _GDeal):
            return CHANCE
        if isinstance(s, (_GTerm, _GDummy)):
            return TERMINAL
        return _BASE.current_player(s)

    def is_terminal(self, s):
        if isinstance(s, (_GTerm, _GDummy)):
            return True
        if isinstance(s, (_GRoot, _GDeal)):
            return False
        return _BASE.is_terminal(s)

    def legal_actions(self, s):
        if isinstance(s, _GRoot):
            return [_TERM, _FOLLOW]
        return _BASE.legal_actions(s)

    def chance_outcomes(self, s):
        if not isinstance(s, _GDeal):
            return _BASE.chance_outcomes(s)   # the round-2 public-card deal
        h = s.h
        out = []
        total = 0.0
        for hr in range(NUM_CARDS):
            if hr == h:
                continue
            p = float(DEAL[h, hr] * self.reachR[hr])
            if p > 0.0:
                out.append((hr, p))
                total += p
        out.append((_DUMMY, max(0.0, 1.0 - total)))   # remaining mass → dummy
        return out

    def infoset_key(self, s, player):
        if isinstance(s, _GRoot):
            return f"G|R{self.resolver}|h{s.h}"        # opponent's gadget infoset
        return _BASE.infoset_key(s, player)

    def apply(self, s, action):
        if isinstance(s, _GRoot):
            return _GTerm(s.h) if action == _TERM else _GDeal(s.h)
        if isinstance(s, _GDeal):
            if action == _DUMMY:
                return _GDummy()
            hr = action
            if self.resolver == 0:
                c0, c1 = hr, s.h
            else:
                c0, c1 = s.h, hr
            return round2_entry_state(self.contrib, self.bets, c0, c1)
        return _BASE.apply(s, action)

    def returns(self, s):
        if isinstance(s, _GTerm):
            v = float(self.cfv_opp[s.h])
            out = [0.0, 0.0]
            out[self.opp] = v
            out[self.resolver] = -v
            return out
        if isinstance(s, _GDummy):
            return [0.0, 0.0]
        return _BASE.returns(s)


def safe_rebel_agent_profile(round1_profile: Profile, leaf_cfv_fn, *,
                             resolve_iters: int = 800) -> Profile:
    """Full-game profile: trunk round-1 + SAFELY re-solved round 2.

    `leaf_cfv_fn(contrib, bets, r0, r1) -> (2,6)` must be the SAME value function
    the trunk solved against, so the counterfactual values the gadget preserves
    are exactly what round-1 was optimized to deliver — the consistency that makes
    the depth-limited agent non-exploitable.
    """
    from .resolve import leaf_reach, round2_leaf_lines

    agent: Profile = dict(round1_profile)
    for line in round2_leaf_lines():
        r = leaf_reach(round1_profile, line)
        cfv = leaf_cfv_fn(line["contrib"], line["bets"], r[0], r[1])
        strat = safe_resolve_round2(line["contrib"], line["bets"], r[0], r[1],
                                    cfv, iters=resolve_iters)
        agent.update(strat)
    return agent


def safe_resolve_round2(contrib, bets, reach0, reach1, cfv, *, iters: int = 600):
    """Safe round-2 strategy for BOTH players; `cfv` is the (2,6) trunk leaf CFV.

    Runs the gadget once per resolver, keeping each resolver's own round-2 play.
    Returns a Profile over the round-2 infosets on this line.
    """
    reach = (np.asarray(reach0, float), np.asarray(reach1, float))
    cfv = np.asarray(cfv, float)
    out: Profile = {}
    for resolver in (0, 1):
        opp = 1 - resolver
        gadget = SafeResolveGadget(contrib, bets, reach[resolver], cfv[opp],
                                   resolver)
        tree = build_tree(gadget)
        solver = CFRSolver(tree, plus=True)
        solver.run(iters)
        prof = solver.average_profile()
        for key, dist in prof.items():
            if key.startswith("G|"):
                continue
            if tree.infoset_player.get(key) == resolver:
                out[key] = dist
    return out
