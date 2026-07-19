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
  + pot (bb) + **chips behind** (bb) + both players' 169-class belief. Slice-G's
  note ("encode the public line/history in the features") is honoured: board +
  pot + stack + belief *is* the public state at the depth boundary. Chips-behind
  is load-bearing, not decoration — a river subgame with 20bb behind is a
  different game from the same pot with 0 behind, and turn betting drives it
  from 20 to 0. (One scalar is exact: a street only closes once both players
  have matched.)
* **target** — the per-169-class root value of the *river* subgame from an exact
  Slice-D CFR+ solve, **for both players** (a 2×169 stacked target: OOP head and
  IP head). Values are *normalized per-hand EV* (bb per matchup), because the
  features carry a normalized belief — a target scaling with the opponent's
  absolute reach mass would not be a function of the input. Off-blueprint
  beliefs are sampled (random, occasionally strength-tilted, sparse) per
  Slice-G's note.

Rows are sampled at the `(pot, chips-behind)` states a turn subgame can actually
reach at its river deal — derived from the tree itself (`river_entry_states`), so
it tracks the bet grid automatically and the net is trained on the distribution
it will be *queried* at.

Data is SpotKey-stratified (flop iso-class recorded per row) and written to
Parquet via pyarrow.

## The depth-limited solver

`DepthLimitedTurnSolver` is `SubgameSolver` with the river chance node replaced
by a **net leaf**. The leaf mirrors exactly what the chance node it replaces
would do: for each river card it queries the net at the post-river belief,
converts the prediction into the opponent-reach-weighted **counterfactual units**
`_walk` propagates, masks the combos that river kills, and divides by the chance
node's divisor. Only training (`_walk`) is depth-limited; **exploitability is
measured in the FULL turn+river game** — the net-driven turn strategy grafted
onto the exact river continuation, best-responded with Slice-D's verified
vectorized BR. So the reported number is the real cost of the net's
approximation, exactly the L4 metric.

A leaf standing in for a subtree has to match it in **units** and in **state**;
this harness got both wrong at first, and the Go/no-go section below records
what that cost. Regression tests now pin all three properties: the leaf's
magnitude against the branch it replaces, an *oracle*-leaf solve reproducing
the full turn+river solve, and the leaf being queried at exactly the tree's
river-entry `(pot, stack)` states.

Per Slice-G's warning, we measure exploitability **against the depth-limited
oracle baseline, not a naively-glued full agent** (that agent's ~0.42 Leduc
ceiling is the equilibrium-selection problem safe continual resolving exists to
fix — explicitly out of toy scope).

## Results

Run: `run_spike(n_rows=3000, n_eval=20, seed=0, gen_iters=80, eval_iters=150,
epochs=400, eval_keep_frac=0.10)` — a representative local-minutes spike
(~14 min on CPU). `run_spike` defaults to ~1e4 rows for a production run.

| metric | value |
|---|---|
| data rows (river solves) | 3000 |
| value-net held-out loss (masked MSE over both heads, bb²) | 117.7 |
| … as RMS CFV error / target std | 10.9 bb / 16.2 bb → **45% of target variance unexplained** |
| held-out eval subgames | 20 |
| mean exploitability — **oracle** (exact turn+river) | **0.191 bb** |
| mean exploitability — **net-driven** depth-limited turn | **3.257 bb** |
| go/no-go threshold | 0.75 bb |
| **verdict** | **NO-GO** (net-driven 3.26 bb ≫ 0.75 bb bar) |

> **Superseded runs**, kept for the record — the verdict was NO-GO in all three:
>
> | run | net-driven | loss | defect present |
> |---|---|---|---|
> | 1st recorded | 6.500 bb | 29.06 | leaf units (net entered CFR 36–62× light) + IP leaf `-v0` |
> | 2nd | 3.530 bb | 28.76 | leaf queried at the subgame root pot, full stack assumed |
> | **current** | **3.257 bb** | **117.7** | — |
>
> **The loss column is not comparable across runs.** Fixing the pot/stack
> mismatch widened the training distribution (pots to ~60bb instead of ≤20bb),
> so the targets themselves are ~3× larger and a larger MSE is expected. The
> scale-free reading — 45% of target variance unexplained — is the one to use.

## Go / no-go

**NO-GO at spike scale.** The mechanism is validated — the exact turn+river
oracle converges to a low **0.19 bb** exploitability baseline, and the harness
measures the net-driven turn strategy in the *full* game with Slice-D's verified
BR — but the value net at 3000-row scale is far too coarse, leaving **45% of
the target CFV variance unexplained**: the net-driven depth-limited turn solve
is **3.26 bb** exploitable, ~17× the oracle and ~4.3× the bar.

**It took three runs to earn that sentence.** The first write-up asserted the
gap was "a data/accuracy result, not a mechanism bug". That was false when
written — there were *two* independent harness bugs, and both had to be fixed
before the claim could be made honestly:

1. **Leaf units.** The net's normalized output was fed straight into `_walk`,
   which propagates opponent-reach-weighted counterfactual values, so the leaf
   branch arrived 36–62× light and CFR effectively ignored it. IP's leaf also
   used `-v0`, assuming a zero-sum-per-matchup subgame that dead money makes
   false.
2. **Leaf state.** The leaf was queried at the subgame *root* pot with a full
   stack implicitly assumed, but a river entry sits wherever the turn betting
   left it — pot 8/16/24/48 with 20/16/12/0 behind on this tree. Chips-behind
   was not a feature at all, so the net could not represent the difference.
   **75% of the states the leaf actually queries fell outside the training
   support.**

Only now, with the leaf verified to enter CFR on the right scale (an oracle leaf
reproduces the exact solve) and queried on-distribution (data-gen samples the
same reachable `(pot, stack)` set the leaf sees), is the remaining gap
attributable to net accuracy. Note the fixes moved the *number* very little
(6.50 → 3.53 → 3.26 bb): the verdict was never in doubt, but the reasoning
behind it was wrong twice.

This is the *expected* L4 outcome and mirrors Slice G's Leduc finding: a value
net drives a depth-limited solver only as well as its CFV accuracy allows, and
at toy data/compute scale that accuracy is nowhere near the ≤1 bb regime a
trustworthy turn strategy needs. Larger runs shrink the loss and the gap, but
closing it to the bar is a research-scale effort — out of toy scope (plan
§7/§9), and nothing downstream depends on it: M4 tier-2 grading uses the *exact
cached solves*, never this value net.

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
