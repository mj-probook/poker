"""Vectorized NLHE environment (impl doc §1, §3 Slice H; plan M6/§6.3).

Approved deviation from the plan's Rust port: numpy batching over the Slice-A
engine. The **port seam is this API + the shared M0 differential** — a future
native env must satisfy both. `VecEnv` is a single-hero RL surface (villains are
a fixed policy, e.g. the §6.1 perturbation bot); `SingleEnvAdapter` drives one
full hand through the discrete encoding so the identical PokerKit differential
validates the env's mechanics/encoding path.

    VecEnv(n, config).reset() -> obs[n, OBS_DIM]
    VecEnv.step(actions[n]) -> (obs[n, OBS_DIM], rewards[n], dones[n])
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from pokerlab.engine.cards import Deck
from pokerlab.engine.state import Hand, HandSetup
from pokerlab.env.encoding import (
    DEFAULT_FRACTIONS,
    OBS_DIM,
    decode_action,
    encode_obs,
    legal_action_mask,
    legal_engine_actions,
    num_actions,
)
from pokerlab.types import Action


@dataclass
class EnvConfig:
    seats: int = 2
    bb: int = 100
    sb: int = 50
    ante: int = 0
    start_stack: int = 10_000  # chips
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS
    hero: int = 0  # the learning/controlled seat


def single_env_action(hand: Hand, rng: random.Random) -> Action:
    """Uniform-random legal action, selected through the discrete env encoding.

    Module-level (picklable) so `diff_harness.run_differential(action_fn=...)`
    replays the identical PokerKit oracle through the vectorized-env path.
    """
    legal = legal_engine_actions(hand)
    return legal[rng.randrange(len(legal))][1]


class SingleEnvAdapter:
    """Names the discrete-encoding policy that reuses the M0 differential."""

    action = staticmethod(single_env_action)


class VecEnv:
    """``n`` independent single-hero NLHE hands stepped in lockstep."""

    def __init__(
        self,
        n: int,
        config: EnvConfig | None = None,
        villain=single_env_action,
        seed: int = 0,
    ) -> None:
        self.n = n
        self.cfg = config or EnvConfig()
        self.villain = villain
        self._seed = seed
        self._hand_count = [0] * n
        self._rng = [random.Random((seed * 1_000_003 + i * 9_176 + 1) & 0xFFFFFFFF) for i in range(n)]
        self.hands: list[Hand] = [None] * n  # type: ignore[list-item]
        self._pending = [0.0] * n
        self.hands_dealt = 0

    # ---- lifecycle -------------------------------------------------------
    def _deal(self, i: int) -> Hand:
        hc = self._hand_count[i]
        self._hand_count[i] += 1
        self.hands_dealt += 1
        seed = ((self._seed * 1_000_003 + i * 10_007 + hc) * 2654435761) & 0xFFFFFFFF
        deck = Deck(seed)
        seats = self.cfg.seats
        cards = deck.deal(2 * seats + 5)
        hole = tuple((cards[2 * k], cards[2 * k + 1]) for k in range(seats))
        board = tuple(cards[2 * seats : 2 * seats + 5])
        setup = HandSetup(
            stacks=tuple([self.cfg.start_stack] * seats),
            button=hc % seats,  # rotate the button each hand
            bb=self.cfg.bb,
            sb=self.cfg.sb,
            ante=self.cfg.ante,
            hole=hole,
            board=board,
        )
        return Hand(setup)

    def _advance_villains(self, i: int) -> None:
        h = self.hands[i]
        while not h.is_terminal() and h.to_act is not None and h.to_act != self.cfg.hero:
            h.apply(self.villain(h, self._rng[i]))

    def _hero_reward(self, i: int) -> float:
        return float(self.hands[i].payoffs()[self.cfg.hero])

    def _fresh_until_hero(self, i: int) -> float:
        """Deal until the hero has a decision; bank reward from skipped hands."""
        acc = 0.0
        for _ in range(10_000):
            self.hands[i] = self._deal(i)
            self._advance_villains(i)
            if self.hands[i].is_terminal():
                acc += self._hero_reward(i)
                continue
            return acc
        raise RuntimeError("hero never reaches a decision (degenerate config)")

    def reset(self) -> np.ndarray:
        obs = np.zeros((self.n, OBS_DIM), dtype=np.float32)
        for i in range(self.n):
            self._pending[i] = self._fresh_until_hero(i)
            obs[i] = encode_obs(self.hands[i])
        return obs

    def step(self, actions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(actions) != self.n:
            raise ValueError(f"expected {self.n} actions, got {len(actions)}")
        obs = np.zeros((self.n, OBS_DIM), dtype=np.float32)
        rewards = np.zeros(self.n, dtype=np.float32)
        dones = np.zeros(self.n, dtype=bool)
        for i in range(self.n):
            h = self.hands[i]
            h.apply(decode_action(h, int(actions[i]), self.cfg.fractions))
            self._advance_villains(i)
            r = self._pending[i]
            self._pending[i] = 0.0
            if h.is_terminal():
                r += self._hero_reward(i)
                dones[i] = True
                self._pending[i] = self._fresh_until_hero(i)  # auto-reset
            rewards[i] = r
            obs[i] = encode_obs(self.hands[i])
        return obs, rewards, dones

    def legal_masks(self) -> np.ndarray:
        """Boolean mask [n, num_actions] for each lane's current hero decision."""
        m = np.zeros((self.n, num_actions(self.cfg.fractions)), dtype=bool)
        for i in range(self.n):
            m[i] = legal_action_mask(self.hands[i], self.cfg.fractions)
        return m
