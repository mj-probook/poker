# Unsafe subgame resolving — Slice G (M5 toy ReBeL on Leduc)

> A learning-goal writeup (plan Goal-A / §7 L3). Building the depth-limited ReBeL
> mechanism on Leduc surfaced, experimentally, the core obstacle the CFR-D /
> DeepStack / ReBeL literature exists to solve. This note records what broke, the
> numbers, and the fix — because reproducing the trap yourself is the point.

All numbers are on 2-player Leduc hold'em, NashConv in chips (OpenSpiel units),
measured by our own `cfr.nash_conv`. The full-game CFR+ equilibrium σ* has
**NashConv ≈ 4–7e-4** (our M1 bar).

## The setup

Depth-limit Leduc at the round-1 / round-2 boundary. The round-1 *trunk* is
solved by CFR; at each round-2-entry leaf a *value function* supplies the
continuation counterfactual values (CFVs). Round 2 is the last street (real
terminals), so a round-2 subgame at a fixed belief solves exactly.

The agent's full strategy = trunk round-1 + a round-2 strategy obtained by
re-solving each round-2 subgame at the belief the trunk produced.

## Discovery 1 — naive/decomposed round-2 re-solving is UNSAFE

Take σ*'s *own* (genuine-equilibrium) round-1. Re-solve each round-2 subgame in
isolation at the belief σ* induces, and glue the results on:

| agent | NashConv |
|---|---|
| σ* (whole) | ~3.8e-4 |
| σ* round-1 + **isolated** round-2 re-solves | **0.21** |

The re-solves are themselves converged subgame equilibria (subgame NashConv
≤1e-3) — they are just the *wrong* equilibrium. A round-2 subgame has many
equilibria of equal value; play at hands that reach the leaf with low probability
under the blueprint is essentially unconstrained, and a full-game best-responder
**deviates in round 1** to arrive with an off-blueprint range and punish it. This
is the textbook unsafe-resolving failure.

## Discovery 2 — the fix, and the detail that is the whole ballgame

The CFR-D re-solving *gadget* makes re-solving safe: the opponent, per private
hand, may TERMINATE for a fixed payoff equal to the counterfactual value the
blueprint promised them, or FOLLOW into the subgame. In equilibrium the resolver
is forced to hold the opponent to that value for every hand.

It only works if you preserve the **right** values:

| gadget CFV target (σ* round-1) | NashConv |
|---|---|
| isolated re-solve's equilibrium value | 0.21 (no better) |
| **blueprint's realized continuation value** (σ* continuation) | **2.7e-4** |

The isolated subgame's equilibrium value and the value σ* *actually realizes* at
the same belief differ at low-reach infosets — because σ*'s round-2 play is
supported by full-game round-1 incentives, not by the isolated subgame. Preserve
what the blueprint realizes, not what a fresh solve prefers.

## Discovery 3 — depth-limited trunks pick SPURIOUS round-1 equilibria

Freezing the continuation to a fixed blueprint reproduces the full game's root
*value* (our depth-limited-oracle test passes to <2e-3), but its round-1
equilibria are not unique — and some are **not extendable** to a non-exploitable
full agent. The trunk's σ1 can differ from σ*'s round-1 by up to 0.34 per action
while achieving the same root value, and no round-2 completion (safe or not) makes
it safe:

| agent | NashConv |
|---|---|
| net-driven trunk round-1 + safe gadget (accurate net, exact gadget) | **~0.42** |

This is not a value-accuracy problem: the value net is essentially exact
(below), and the gadget is exact. It is the equilibrium-selection problem that
DeepStack-style **safe continual resolving** (maintaining CFV consistency at every
re-solve, not just at the leaves) exists to solve. That is out of toy scope
(plan §9); nothing downstream depends on it.

## What DOES land (Slice G exits)

With the value function = σ* continuation (the "leaf values from the full CFR+
solution"), sampled at reach-perturbed off-blueprint beliefs per line:

- **Value net**: 30k self-generated PBS→CFV samples (~6s), tiny MLP, CPU/seeded,
  held-out **RMSE ≈ 7e-4** (target std ≈ 8e-2).
- **Net-driven depth-limited trunk reproduces the full-game root value to
  |Δ| ≈ 0** (≤2e-3) — the net is an accurate drop-in for the oracle leaves.
  *Key implementation detail:* the public **betting line** (not just the pot) is
  part of the PBS and must be in the net's features — σ*'s round-2 play is
  history-indexed, so two lines sharing a pot have different continuation values.
- **Oracle-leaf + safe-resolve** (genuine round-1 + blueprint CFVs + gadget)
  reaches **NashConv ≈ 1.1e-3 ≤ 2e-3** — the M5 bar.

Reproduce all of it: `uv run python scripts/train_leduc_valuenet.py`.

## Notes for Slice I (tiny-HUNL spike)

Reuse the **trunk + value-net pattern**, not the naive agent glue:

1. `rebel.trunk.DepthLimitedSolver` + a `leaf_value_fn` — the belief-vectorized
   depth-limited CFR generalizes to more streets (it is street-agnostic; only the
   leaf value fn and the public tree change).
2. `rebel.valuenet` / `rebel.dataset` — train a PBS→CFV net from self-generated
   solver labels; **encode the public line/history in the features**, sample
   off-blueprint beliefs.
3. `rebel.safe_resolve` — use the gadget with **blueprint** CFVs for any re-solve.
4. Expect the same spurious-round-1 ceiling on full-agent exploitability unless
   safe continual resolving is added; measure exploitability against the
   depth-limited baseline (the L4 go/no-go), not against a naive glued agent.
