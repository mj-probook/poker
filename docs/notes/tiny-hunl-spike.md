# Tiny-HUNL depth-limited value-net spike — Slice I (M7 / plan §7 L4)

> Learning-goal writeup (Goal-A, L4). Carries the M5 ReBeL pattern
> (`rebel.trunk`) from Leduc to no-limit hold'em turn/river, reusing the Slice-D
> subgame solver as both the exact oracle and the CFV labeller. Records the
> go/no-go metric and the numbers. Everything is local (numpy + a small torch
> MLP); **no cloud spend**.

Code: `pokerlab.spike.hunl`. Reproduce: `pokerlab.spike.hunl.run_spike(...)`.

## Setup

Tiny-HUNL config: **20bb** effective, bet menu **{0.5 pot, 1.0 pot, jam}**,
turn/river subgames only (never a full flop — a flop solve is out of scope,
plan §3/§9). OOP acts first; the hero is the OOP root actor.

The value net is the classic PBS→CFV river network:

* **features (the PBS, public line/history encoded)** — 52-card board multi-hot
  + pot (bb) + both players' 169-class belief. Slice-G's note ("encode the
  public line/history in the features") is honoured: the board + pot + belief
  *is* the public state at the depth boundary.
* **target** — the OOP per-169-class root counterfactual value of the *river*
  subgame, from an exact Slice-D CFR+ solve. Off-blueprint beliefs are sampled
  (random, occasionally strength-tilted, sparse) per Slice-G's note.

Data is SpotKey-stratified (flop iso-class recorded per row) and written to
Parquet via pyarrow.

## The depth-limited solver

`DepthLimitedTurnSolver` is `SubgameSolver` with the river chance node replaced
by a **net leaf**: at a river-entry it averages the net's river CFV over the
runout (44 cards) and returns per-combo values (IP = zero-sum complement, a
tiny-spike approximation whose error the metric then captures). Only training
(`_walk`) is depth-limited; **exploitability is measured in the FULL turn+river
game** — the net-driven turn strategy grafted onto the exact river continuation,
best-responded with Slice-D's verified vectorized BR. So the reported number is
the real cost of the net's approximation, exactly the L4 metric.

Per Slice-G's warning, we measure exploitability **against the depth-limited
oracle baseline, not a naively-glued full agent** (that agent's ~0.42 Leduc
ceiling is the equilibrium-selection problem safe continual resolving exists to
fix — explicitly out of toy scope).

## Results

Run: `run_spike(n_rows=3000, n_eval=20, seed=0, gen_iters=80, eval_iters=150,
epochs=400, eval_keep_frac=0.10)` — a representative local-minutes spike
(~12 min on CPU). `run_spike` defaults to ~1e4 rows for a production run.

| metric | value |
|---|---|
| data rows (river solves) | 3000 |
| value-net held-out loss (masked MSE, bb²) | 29.06 |
| held-out eval subgames | 20 |
| mean exploitability — **oracle** (exact turn+river) | **0.191 bb** |
| mean exploitability — **net-driven** depth-limited turn | **6.500 bb** |
| go/no-go threshold | 0.75 bb |
| **verdict** | **NO-GO** (net-driven 6.50 bb ≫ 0.75 bb bar) |

## Go / no-go

**NO-GO at spike scale.** The mechanism is validated — the exact turn+river
oracle converges to a low **0.19 bb** exploitability baseline, and the harness
measures the net-driven turn strategy in the *full* game with Slice-D's verified
BR — but the value net at 3000-row scale (held-out loss **29 bb²**, i.e. an RMS
CFV error of ~5 bb) is far too coarse: the net-driven depth-limited turn solve
is **6.5 bb** exploitable, ~34× the oracle and ~9× the bar.

This is the *expected* L4 outcome and mirrors Slice G's Leduc finding: a value
net drives a depth-limited solver only as well as its CFV accuracy allows, and
at toy data/compute scale that accuracy is nowhere near the ≤1 bb regime a
trustworthy turn strategy needs. The gap is a **data/accuracy** result (the
net), not a mechanism bug (the oracle and BR are exact). Larger runs shrink the
loss and the gap, but closing it to the bar is a research-scale effort — out of
toy scope (plan §7/§9), and nothing downstream depends on it: M4 tier-2 grading
uses the *exact cached solves*, never this value net.

## What lands / what's out of scope

* **Lands:** the whole pipeline runs locally end-to-end (data-gen → Parquet →
  net → depth-limited eval), the mechanism is validated (oracle turn+river
  converges to a low-exploitability baseline; the net-driven solver is measured
  against it in the full game with the verified BR), and the go/no-go is
  reproducible from one call.
* **Out of scope (plan §9):** safe *continual* resolving (CFV consistency at
  every re-solve, not just leaves) — without it the same equilibrium-selection
  ceiling Slice G documented on Leduc applies to a full multi-street agent.
  Nothing downstream depends on it; the M4 grading pipeline uses the exact
  cached solves (tier 2), not the value net.
* Larger/production runs: `run_spike` defaults to ~1e4 rows; net accuracy (and
  thus the net-driven exploitability) improves with dataset size — the recorded
  run is a representative local-minutes spike, not a converged research result.
