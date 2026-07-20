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

Boards are drawn i.i.d.; each row RECORDS its flop iso-class. That is not
stratification, which is what this line used to claim (wave-3 [M9]) — no
mechanism balances rows across iso-classes, and a 400-draw sample lands in 350
distinct classes with 1–4 rows each. Written to Parquet via pyarrow.

## The depth-limited solver

`DepthLimitedTurnSolver` is `SubgameSolver` with the river chance node replaced
by a **net leaf**. The leaf mirrors exactly what the chance node it replaces
would do: for each river card it queries the net at the post-river belief,
converts the prediction into the opponent-reach-weighted **counterfactual units**
`_walk` propagates, masks the combos that river kills, and divides by the chance
node's divisor. Only training (`_walk`) is depth-limited; **exploitability is
measured in the FULL turn+river game** — the net-driven turn strategy grafted
onto the exact river continuation, best-responded with Slice-D's verified
vectorized BR. So the reported number is an **upper bound** on the cost of the
net's approximation, which is the L4 metric. It is not purely that cost: the
measured exploitability also contains the residual from finite CFR iterations
and from the depth-limit itself, neither of which is attributable to the net.
Calling it "the real cost of the net's approximation" claimed an attribution
the harness cannot make (wave-3 [M7]).

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
| … as RMS CFV error / target std | 10.85 bb / 15.04 bb → **52% of target variance unexplained** |
| held-out eval subgames | 20 |
| mean exploitability — **oracle** (exact turn+river) | **0.191 bb** |
| mean exploitability — **net-driven** depth-limited turn | **3.257 bb** |
| net / oracle ratio | **17.1×** |
| go/no-go rule | net ≤ oracle × tolerance (tolerance 2.0) |
| **verdict** | **NO-GO** (17.1× the oracle, against a 2× tolerance) |

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
> scale-free reading — 52% of target variance unexplained — is the one to use.

**Std convention (wave-3 [M6]).** "Target variance" means the MEAN-CENTRED
variance of the target CFVs over masked entries only (20.0% of the target
matrix; unmasked entries are not predictions and must not dilute it), i.e. the
usual 1 − R². On the recorded 3000-row dataset that std is **15.04 bb**, so
`117.7 bb² / 15.04² = 52%`.

This convention has to be stated because it dominates the answer: measured about
zero instead of about the mean, the same numbers read **23%** unexplained,
because the targets have a large positive mean (16.79 bb) and pot-sized offsets
would masquerade as explained variance. The previously recorded **45%** implied
a std of 16.2 bb, which does not reproduce under either convention; it is
corrected here rather than carried forward.

## Go / no-go

**NO-GO at spike scale.** The mechanism is validated — the exact turn+river
oracle converges to a low **0.19 bb** exploitability baseline, and the harness
measures the net-driven turn strategy in the *full* game with Slice-D's verified
BR. The net-driven depth-limited turn solve is **3.26 bb** exploitable,
**17.1× the oracle**. The held-out CFV loss leaves **52% of the target variance
unexplained**.

Those are two measurements, stated side by side. What this note previously did
was join them with "because", and that link is now falsified — see below.

**The go rule is now relative (wave-3 [M3]).** It used to be an absolute
`≤ 0.75 bb`, a constant with no derivation behind it, compared against nothing —
even though the baseline it should have been measured against was already being
computed in the same loop: the exploitability of the SAME depth-limited
decomposition with an exact (solved) leaf. That is the floor this design can
reach, so the question is "how much worse than the best this decomposition can
do", not "how many big blinds". The ratio is also the scale-free number: the bb
figure moves with whatever pot sizes the eval happens to draw, the ratio does
not. The verdict is unchanged and robust either way — 17× is not near any
plausible tolerance.

### The causal chain this note used to assert is FALSIFIED (wave-3 [M1])

The claim was: the net's held-out CFV loss is too high, and *therefore* the
net-driven solver is exploitable. A pre-registered contrast tested it directly
by training a net with substantially LOWER held-out loss and measuring the
exploitability it produced.

| training regime | held-out CFV loss | net-driven exploitability |
|---|---|---|
| as shipped (full-batch), seed 0 | 117.72 | **3.257 bb** (oracle 0.191) |
| lower-loss (minibatch), seed 0 | 59.48 (−49%) | **4.028 bb** (+23.7% WORSE) |
| as shipped (full-batch), seed 1 | 121.94 | **3.330 bb** (oracle 0.154) |
| lower-loss (minibatch), seed 1 | 55.84 (−54%) | **3.919 bb** (+17.7% WORSE) |

Halving the loss made the strategy *worse*, on two independent draws. So the
"because" was never established: it was a plausible story fitted to a single
run, and it is the third defect of this exact shape in this document's history
(see the two below).

The training loop is **full-batch** — one gradient step per epoch — which is a
real limitation (wave-3 [M5]) and is what the lower-loss arm changes. Note the
as-shipped loss is stable across seeds (117.72 / 121.94): the recorded *number*
was always sound. It is the "because" attached to it that failed.

Three guards on how far that result may be pushed:

* **Two seeds buy independence on the data axis only.** This is "not
  dataset-specific". It is **not** "not an artifact" — both arms share one
  evaluation protocol, and the belief-support gap below remains a live
  candidate artifact.
* **n = 20 held-out subgames per arm, and no dispersion was measured.** This is
  real evidence, not a significance claim.
* The safe statement, and no stronger one: **held-out CFV loss is not a
  reliable proxy for net-driven exploitability in this harness.**

The *measurements* in this note stand and reproduce. What was wrong was the
mechanism attached to them. The NO-GO verdict is unaffected — every arm
measured is 17–26× the oracle — and is robust either way.

### The most mechanical candidate explanation: the belief axis (wave-3 [M2]) — tested in R4 and rejected

Offered as a candidate in wave 3; measured in round 4; excluded.

The net is queried, inside CFR, on beliefs it was never trained on — and not
merely shifted, but largely disjoint:

* **65.7% of leaf queries fall entirely outside the training belief support**
  (training ranges span 14–53 non-zero classes; queries span 0–29). R4
  measurement: on every queried belief where an exact CFV exists (n=200 exact
  `solve_river`), the net's masked MSE is 127.99 mean / 120.52 median vs 117.72
  held-out control — **1.09×**. Off-distribution in the sparsity sense did not
  translate into wrong. Candidate excluded.
* **45% of queries are all-zero belief vectors**, which occur zero times in
  training. That sentence was corrected twice in R4, both times by its own
  author against their own claim: (1) each head is multiplied by the
  OPPONENT's valid reach (`counterfactual_from_normalized`), so an
  opponent-dead prediction is annihilated by exact zero and never reaches CFR.
  The R4 per-branch decomposition (n=9504): both-live 24.2%, OOP-dead-only
  37.9%, IP-dead-only 0.0%, both-dead 37.9% — so injection touches ONE head
  (OOP) on 37.9% of queries, and the IP head is never injected at all;
  own-reach-free values are correct CFR semantics where it does occur. (2) the
  surviving half is measured negligible — zeroing 90.3% of OOP-head and 35.7%
  of IP-head leaf contributions moved exploitability by **−0.3%** (3.257 →
  3.248 bb). Structural reason, in hindsight: regrets on combos the player
  cannot hold are non-binding, because CFR weights the played strategy by own
  reach.

This had two halves; [R4-c] closed the parameter half, the structural half
stays open:

* **The parameter half.** Evaluation draws sparser ranges than data generation.
  Two figures, because the recorded run and the shipped default are not the same
  configuration — measured as mean non-zero classes per sampled range, seed 0,
  n=250:

  | configuration | eval density | mean non-zero classes | vs generation (0.2 → 34.2) |
  |---|---|---|---|
  | the run recorded above, and now the shipped default | 0.10 | **17.1** | half as dense |

  These used to be two rows. The recorded run passed `eval_keep_frac=0.10`
  explicitly while `EVAL_KEEP_FRAC` shipped 0.12 (**20.6** classes), so the note
  and the code described different configurations and two independent
  measurements of "the" sparsity gap disagreed (17.1 vs 20.6) with neither being
  wrong. [R4-c] aligned the default DOWN to the density that produced every
  number here, which removes the ambiguity at its source rather than annotating
  it. Nothing was re-measured: the figures are the ones already recorded, now
  describing the configuration that actually ships.

  This ambiguity is now CLOSED, and the direction matters. [R4-c] aligned the
  DEFAULT DOWN to the recorded density, so the table's row becomes the shipping
  configuration and every number in this note — and both of [M1]'s seed
  measurements — still describes what the code draws. Nothing went stale;
  that is why this alignment was safe and the other one is not.

  Aligning EVAL to GEN would still leave the results table describing
  a configuration that no longer ships — the recorded-vs-landed failure in
  reverse, and the more expensive mistake — so that alignment stays refused and
  `test_the_two_densities_are_not_silently_equalised` still guards it. Aligning
  the DEFAULT to the recorded run is the opposite move: it makes the shipped
  configuration match the numbers rather than making the numbers stale.
  `test_spike_hunl` pins the figures against the code so neither can move
  without this note moving too.
* **The structural half, which no parameter closes.** Beliefs reaching the leaf
  are *strategy-weighted reaches* produced by CFR iterations; training beliefs
  are sampled independently. No value of any density parameter makes an
  independently-sampled range look like a reach vector a solver walked to.
  Closing it means generating training data along the solver's own trajectory —
  the ReBeL self-play loop. That is a design change, not a tuning one, and it is
  recorded here as open rather than papered over.

**R4 end state:** the 17× gap is **unattributed**, with two candidates
positively excluded (belief-axis-as-inaccuracy at 1.09×; degenerate-query
injection at −0.3%). Remaining candidates, unmeasured and recorded:
class-level aggregation (one value per 169-class; combos within a class
differ) and the grafted-river artifact above. The 3.257 bb net-driven
baseline has reproduced three independent times (R3 original, R3 seed-0
re-run, R4 arm A). Incidental performance note, not correctness: 75.8% of
eval-loop net inference (7200/9504 queries) is degenerate states, the IP
head's degenerate contributions all multiplied by zero on arrival.

### Two harness bugs preceded all of this

The first write-up asserted the gap was "a data/accuracy result, not a mechanism
bug". That was false when written — there were *two* independent harness bugs,
and both had to be fixed before any claim could be made honestly:

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

The leaf now enters CFR on the right scale (an oracle leaf reproduces the exact
solve) and is queried at the reachable `(pot, stack)` states data-gen samples.
The fixes moved the *number* very little (6.50 → 3.53 → 3.26 bb): the verdict
was never in doubt, but the reasoning behind it was wrong — and, per [M1] above,
the third attempt at a mechanism was wrong too. **The remaining gap is
unattributed.** Net accuracy is no longer a supportable explanation for it,
because lowering the loss made the strategy worse.

The NO-GO itself is the expected L4 outcome: a depth-limited solver on a learned
leaf, at toy data and compute scale, lands nowhere near the regime a trustworthy
turn strategy needs. What this note can no longer say is *why* — "larger runs
shrink the loss and therefore the gap" is precisely the inference [M1]
falsified. Scaling may still close it; this spike provides no evidence that it
would, and the belief-axis gap above is the more mechanical candidate.

Nothing downstream depends on the answer: M4 tier-2 grading uses the *exact
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
