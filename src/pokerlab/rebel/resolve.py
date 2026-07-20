"""Round-2 subgame re-solving for Leduc ReBeL (plan §7 L3; impl doc §3 Slice G).

Round 2 is the last betting street, so its subgames terminate in real showdowns:
given a belief (the ranges entering round 2) they solve *exactly* with the
existing CFR+ engine. Two uses:

  1. **Value oracle** — `resolve_round2(...).cfv` is the per-card counterfactual
     value at a public belief state, the quantity the value net learns to predict.
  2. **Safe depth-limited play** — a depth-limited agent must not paste a
     blueprint's round-2 strategy onto a round-1 line it never trained for (the
     off-belief trap that makes the naive combination exploitable). Instead
     `rebel_agent_profile` re-solves round 2 for the belief the trunk produced.

`round2_leaf_lines()` enumerates the five round-1 lines that reach round 2. The
counterfactual-value convention matches trunk CFR: cfv_p[c] = Σ_{c'} DEAL[c,c']
· r_opp[c'] · U_p(c, c'), where U_p is the equilibrium continuation value with
concrete cards and DEAL is the 1/30 deal prior (own reach excluded).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pokerlab.cfr.cfr import CFRSolver
from pokerlab.cfr.game import Game, Profile, build_tree
from pokerlab.cfr.leduc import CALL, FOLD, RAISE, LeducPoker, LeducState

from .depth_limited import continuation_value, round2_entry_state
from .pbs import DEAL, NUM_CARDS

_BASE = LeducPoker()


# --------------------------------------------------------------------------- #
# Round-1 public betting: the lines that reach round 2 (card-independent).
# --------------------------------------------------------------------------- #
def round2_leaf_lines() -> list[dict]:
    """Enumerate round-1 lines closing into round 2.

    Each entry: {'contrib', 'bets', 'decisions'} where decisions is the ordered
    list of (acting_player, bets_prefix, action) needed to reconstruct reaches.
    """
    out: list[dict] = []

    def walk(s: LeducState, decisions: list) -> None:
        for a in _BASE.legal_actions(s):
            ns = _BASE.apply(s, a)
            rec = decisions + [(s.to_move, s.bets, a)]
            if ns.done:                      # a fold ended round 1 — no round 2
                continue
            if ns.rnd == 2 and ns.public is None:   # round 1 closed → leaf
                out.append({"contrib": ns.contrib, "bets": ns.bets,
                            "decisions": rec})
            else:
                walk(ns, rec)

    # Dummy distinct private cards bypass the deal; betting ignores card identity.
    walk(LeducState(private=(0, 1)), [])
    return out


def leaf_reach(round1_profile: Profile, line: dict) -> np.ndarray:
    """Reach (range) each player carries into a round-2 leaf under `round1_profile`."""
    r = np.ones((2, NUM_CARDS))
    for player, bets_prefix, action in line["decisions"]:
        for c in range(NUM_CARDS):
            key = f"P{c}|BNone|{bets_prefix}"
            r[player, c] *= round1_profile[key][action]
    return r


# --------------------------------------------------------------------------- #
# The round-2 subgame as a Game (belief-weighted roots, then public deal).
# --------------------------------------------------------------------------- #
def _blocked(public_card: int | None) -> np.ndarray:
    mask = ~np.eye(NUM_CARDS, dtype=bool)
    if public_card is not None:
        mask[public_card, :] = False
        mask[:, public_card] = False
    return mask


def joint_belief(reach0: np.ndarray, reach1: np.ndarray) -> np.ndarray:
    """Normalized joint over (c0, c1) from ranges entering round 2 (public unseen)."""
    j = DEAL * _blocked(None) * np.outer(reach0, reach1)
    s = j.sum()
    return j / s if s > 0.0 else j


class LeducRound2Subgame(Game):
    """Leduc round 2 rooted at a belief: deal public, bet, show down."""

    num_players = 2

    def __init__(self, contrib: tuple[int, int], bets: tuple[int, ...],
                 belief: np.ndarray):
        self.contrib = contrib
        self.bets = bets
        self.belief = belief
        # All 30 distinct pairs are roots so every round-2 infoset exists in the
        # tree (zero-weight roots build the subtree but never move the average).
        self._roots = [
            (round2_entry_state(contrib, bets, c0, c1), float(belief[c0, c1]))
            for c0 in range(NUM_CARDS) for c1 in range(NUM_CARDS) if c0 != c1
        ]

    def initial_states(self):
        return self._roots

    def current_player(self, s):
        return _BASE.current_player(s)

    def chance_outcomes(self, s):
        return _BASE.chance_outcomes(s)

    def legal_actions(self, s):
        return _BASE.legal_actions(s)

    def infoset_key(self, s, player):
        return _BASE.infoset_key(s, player)

    def apply(self, s, action):
        return _BASE.apply(s, action)

    def is_terminal(self, s):
        return _BASE.is_terminal(s)

    def returns(self, s):
        return _BASE.returns(s)


@dataclass
class Round2Solution:
    strategy: Profile         # round-2 infoset strategies (keyed by full bets)
    cfv: np.ndarray           # (2, 6) per-card counterfactual values
    belief: np.ndarray        # (6, 6) normalized joint used at the root
    u0: np.ndarray            # (6, 6) per-pair P0 continuation value under σ2


def resolve_round2(contrib: tuple[int, int], bets: tuple[int, ...],
                   reach0: np.ndarray, reach1: np.ndarray,
                   iters: int = 500) -> Round2Solution:
    """Solve the round-2 subgame at the given belief; return strategy + cfv."""
    reach0 = np.asarray(reach0, dtype=float)
    reach1 = np.asarray(reach1, dtype=float)
    belief = joint_belief(reach0, reach1)

    subgame = LeducRound2Subgame(contrib, bets, belief)
    tree = build_tree(subgame)
    solver = CFRSolver(tree, plus=True)
    solver.run(iters)
    sigma2 = solver.average_profile()

    # Per-card counterfactual values under the resolved equilibrium.
    mask = _blocked(None)
    cfv = np.zeros((2, NUM_CARDS))
    u0 = np.zeros((NUM_CARDS, NUM_CARDS))
    for c0 in range(NUM_CARDS):
        for c1 in range(NUM_CARDS):
            if c0 == c1:
                continue
            u0[c0, c1] = continuation_value(
                _BASE, sigma2, round2_entry_state(contrib, bets, c0, c1), 0)
    # cfv_0[c] = Σ_{c'} DEAL·mask·reach1[c'] · U0(c,c');  U1 = -U0.
    w = DEAL * mask
    cfv[0] = (w * reach1[None, :] * u0).sum(axis=1)
    cfv[1] = (w * reach0[:, None] * (-u0)).sum(axis=0)
    return Round2Solution(strategy=sigma2, cfv=cfv, belief=belief, u0=u0)


def rebel_agent_profile(round1_profile: Profile, *, resolve_iters: int = 500) -> Profile:
    """Full-game profile via NAIVE decomposed re-solving (round-2 at its beliefs).

    This is UNSAFE (see tests/test_rebel_resolve): it re-solves each round-2
    subgame in isolation without preserving counterfactual values, so the result
    is exploitable. Kept as the baseline the safe gadget (safe_resolve.py) fixes.
    """
    agent: Profile = dict(round1_profile)
    for line in round2_leaf_lines():
        r = leaf_reach(round1_profile, line)
        sol = resolve_round2(line["contrib"], line["bets"], r[0], r[1],
                             iters=resolve_iters)
        agent.update(sol.strategy)
    return agent
