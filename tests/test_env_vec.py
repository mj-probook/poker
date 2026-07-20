"""Slice H: VecEnv shape/determinism/auto-reset/throughput (impl doc §3 H)."""

import os
import time

import numpy as np
import pytest

from pokerlab.env.encoding import OBS_DIM, num_actions
from pokerlab.env.vec import EnvConfig, VecEnv


def _call_policy(n):
    return np.ones(n, dtype=int)  # index 1 = check/call, always legal


def test_reset_and_step_shapes() -> None:
    env = VecEnv(8, EnvConfig(seats=2), seed=1)
    obs = env.reset()
    assert obs.shape == (8, OBS_DIM) and obs.dtype == np.float32
    obs, rewards, dones = env.step(_call_policy(8))
    assert obs.shape == (8, OBS_DIM)
    assert rewards.shape == (8,) and rewards.dtype == np.float32
    assert dones.shape == (8,) and dones.dtype == bool


def test_deterministic_under_seed() -> None:
    def rollout(seed):
        env = VecEnv(6, EnvConfig(seats=3), seed=seed)
        env.reset()
        trace = []
        for _ in range(40):
            obs, r, d = env.step(_call_policy(6))
            trace.append((obs.copy(), r.copy(), d.copy()))
        return trace

    a, b, c = rollout(7), rollout(7), rollout(9)
    for (oa, ra, da), (ob, rb, db) in zip(a, b):
        assert (oa == ob).all() and (ra == rb).all() and (da == db).all()
    # a different seed must diverge somewhere
    assert any((ra != rc).any() or (da != dc).any() for (_, ra, da), (_, rc, dc) in zip(a, c))


def test_auto_reset_and_valid_obs() -> None:
    env = VecEnv(16, EnvConfig(seats=2), seed=3)
    env.reset()
    saw_done = False
    for _ in range(60):
        obs, r, d = env.step(_call_policy(16))
        saw_done = saw_done or bool(d.any())
        assert obs.shape == (16, OBS_DIM)
        assert np.isfinite(obs).all() and np.isfinite(r).all()
    assert saw_done  # check/call showdowns terminate and auto-reset


def test_legal_masks_shape_and_call_always_legal() -> None:
    env = VecEnv(5, EnvConfig(seats=2), seed=2)
    env.reset()
    m = env.legal_masks()
    assert m.shape == (5, num_actions())
    assert m[:, 1].all()  # check/call always legal for the acting hero


def test_runs_with_perturbation_bot_villain() -> None:
    # §6.1 bot serves as M6's fixed opponent inside the vec env.
    from pokerlab.env.bot import PerturbationBot

    bot = PerturbationBot(aggression=0.4, loosen=0.3)
    env = VecEnv(8, EnvConfig(seats=2), villain=bot.act, seed=4)
    env.reset()
    for _ in range(30):
        _, r, d = env.step(_call_policy(8))
        assert np.isfinite(r).all() and d.dtype == bool


def test_throughput_smoke() -> None:
    # Load-robust by construction (round-2 finding [E52]). The bar measures how
    # much CPU the vectorized step path *costs*, so it must be read off
    # process_time, not wall time: wall time also charges us for every other
    # process on the box. Measured on this machine, 2x CPU oversubscription
    # degrades the wall-clock number 2.4x but the process_time number only 1.2x.
    # That is a mitigation, not immunity -- a heavily oversubscribed box (a
    # parallel-agent session, where several suites each spawn cpu_count-2
    # differential workers) can still starve it, so a genuinely loaded box
    # SKIPS loudly instead of reporting a red that means nothing.
    #
    # Deliberately NOT slow/bench-gated: per plan Slice H this is a fast-suite
    # behavior, and round-1 finding [2] is the standing lesson that gating a
    # milestone assertion into a suite nobody runs empties it silently.
    if os.getloadavg()[0] > 2 * (os.cpu_count() or 1):
        pytest.skip(f"box load {os.getloadavg()[0]:.1f} too high to time throughput")
    env = VecEnv(64, EnvConfig(seats=6), seed=5)
    env.reset()
    t0 = time.process_time()
    for _ in range(300):
        env.step(_call_policy(64))
    elapsed = time.process_time() - t0
    hands_per_min = env.hands_dealt / elapsed * 60
    assert hands_per_min >= 50_000, f"only {hands_per_min:.0f} hands/min"
