"""Leduc public belief state (plan §7 L3; impl doc §3 Slice G).

A public belief state (PBS) is what a depth-limited re-solver reasons over: the
*public* observation both players share — public card, betting line, pot — plus
each player's **reach** vector over their private cards (the range). Private
holdings stay hidden; only the reach probabilities are public knowledge once a
strategy is fixed.

Leduc deck: card 0..5, rank = card // 2 (0=J, 1=Q, 2=K), two suits each. The
joint belief over (P0 card, P1 card) is the deal prior (uniform over the 30
distinct ordered pairs) times both reach vectors, renormalized with card removal
(a player cannot hold the public card or the other player's card).

The feature vector (`features()`) is the value net's input: a 3-way one-hot over
the round-2-entry pot levels {2, 6, 10} plus both reach vectors (2×6). The pot
alone identifies the continuation game structure; the reaches carry the belief.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NUM_CARDS = 6
RANKS = ("J", "Q", "K")
# Pot sizes at which round-1 betting closes into round 2 (contribs 1/1, 3/3, 5/5).
POT_LEVELS = (2, 6, 10)
# The five round-1 betting lines that reach round 2, by public betting history.
# The line — not just the pot — is part of the public state: σ*'s round-2 play is
# indexed by the full history, so the value function differs across lines that
# share a pot. The net must see the line, hence it is one-hot in the features.
LEAF_LINES = (
    (1, 1),          # check-check                (pot 2)
    (2, 1),          # raise-call                 (pot 6)
    (1, 2, 1),       # check-raise-call           (pot 6)
    (2, 2, 1),       # raise-raise-call           (pot 10)
    (1, 2, 2, 1),    # check-raise-raise-call     (pot 10)
)
_LINE_INDEX = {b: i for i, b in enumerate(LEAF_LINES)}


def _deal_prior() -> np.ndarray:
    """Joint prior over (c0, c1): uniform over the 30 distinct ordered pairs."""
    w = 1.0 - np.eye(NUM_CARDS)
    return w / w.sum()


DEAL = _deal_prior()


@dataclass
class PBS:
    """Public observation + per-player reach vectors over the 6 private cards."""

    public_card: int | None = None
    bets: tuple[int, ...] = ()
    contrib: tuple[int, int] = (1, 1)
    reach: np.ndarray = field(default_factory=lambda: np.ones((2, NUM_CARDS)))

    @classmethod
    def initial(cls) -> PBS:
        """Root PBS: no public card, antes posted, every hand fully live."""
        return cls()

    # -- belief math --------------------------------------------------------- #
    def _allowed(self) -> np.ndarray:
        """Boolean (6,6): (c0, c1) pairs not ruled out by card removal."""
        mask = ~np.eye(NUM_CARDS, dtype=bool)
        if self.public_card is not None:
            mask[self.public_card, :] = False
            mask[:, self.public_card] = False
        return mask

    def joint_belief(self) -> np.ndarray:
        """Normalized posterior over (P0 card, P1 card); sums to 1 (0 if dead)."""
        j = DEAL * self._allowed() * np.outer(self.reach[0], self.reach[1])
        s = j.sum()
        return j / s if s > 0.0 else j

    def marginal(self, player: int) -> np.ndarray:
        """Normalized belief over `player`'s own card."""
        j = self.joint_belief()
        return j.sum(axis=1) if player == 0 else j.sum(axis=0)

    # -- Bayes update -------------------------------------------------------- #
    def update(self, player: int, action_probs: np.ndarray) -> PBS:
        """Return a new PBS after `player` took an action: reach *= P(action|card).

        `action_probs[c]` is the probability the player takes the observed action
        holding card c. Multiplying the reach in is the exact Bayes step — the
        joint belief renormalizes lazily in `joint_belief()`.
        """
        new_reach = self.reach.copy()
        new_reach[player] = new_reach[player] * np.asarray(action_probs, dtype=float)
        return PBS(self.public_card, self.bets, self.contrib, new_reach)

    def with_reach(self, reach: np.ndarray) -> PBS:
        return PBS(self.public_card, self.bets, self.contrib,
                   np.asarray(reach, dtype=float).reshape(2, NUM_CARDS))

    # -- value-net features -------------------------------------------------- #
    def features(self) -> np.ndarray:
        """One-hot betting line (5) + both reach vectors (12) = 17 dims."""
        onehot = np.zeros(len(LEAF_LINES))
        idx = _LINE_INDEX.get(tuple(self.bets))
        if idx is not None:
            onehot[idx] = 1.0
        return np.concatenate([onehot, self.reach[0], self.reach[1]])
