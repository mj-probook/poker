# Build Plan: Personal Poker Lab — Learn Game Theory + RL, Train for a WSOP Bracelet

> **Rev 3.4 — 2026-08-01 (post-ship build-out reconciliation).** The 2026-07-29/08-01 autonomous build waves closed four §5.1 deferral rows (resteal, ante-adjusted full-ring open/defend — now with a full 2–9-player table-size axis, final-table ladders, postflop tier-2 river drills), shipped the §M2 range grid (its recorded trigger — range-shaped drill answers — arrived with the open game), added tier-3 MULTIWAY drills consuming the §5.2 population table as their grading baseline (frequency flags only, the §5.3 hard rule), and made `drill_attempts.correct` nullable (tier 3 makes no right/wrong claim — the ev_loss NULL rule extended to accuracy). Rows below updated in place with *built (rev 3.4)* markers; per-slice records in the impl doc. Scope and rationale unchanged.
>
> **Rev 3.3 — 2026-07-20 (R4 plan-vs-built audit reconciliation).** The round-4 fresh-eyes audit inventoried 51 plan promises against the build (32 delivered, 6 recorded deviations, 5 recorded deferrals, 8 unrecorded gaps) and this rev records the document's share of those gaps where the build is right and the text was stale or silent: §3 storage lists the two build-added tables (`failed_hands`, `batch_queue`); §3b #2 records the two unbuilt Solution adapters as deferred; §5.2 gets the same as-built honesty table §5.1 got in rev 3.2; §8 M0 records the odd-chip carve-out, M1.5 the measured exploitability + tightened CI bar, M3 the accepted TexasSolver-bars deferral (correcting a "never silently passed" phrasing that promised an assertion deliberately deferred in round 1), M4 the overnight-equals-user-cron deployment note. Code-side R4 findings (tier-2 provenance threading, session-partial scoping, provenance label map, saturation pin) landed as fixes, not plan edits. Scope and rationale unchanged.
>
> **Rev 3.1 — 2026-07-19 (as-built reconciliation).** Implementation review round 2 found that three round-1 findings' plan-restatement halves never landed. This rev restates the M1.5/M3/M4/M5 exit rows (§8) to what the built test suite actually asserts, records the numpy-for-Rust deviation (§3, §8), and adds the depth bucket to the leak-taxonomy key (§5.3). Scope and rationale unchanged.
>
> **Rev 3 — 2026-07-19.** Round-2 review (implementability bar: "could /featuret build this without guessing?") by three reviewers — implementation-readiness, architecture, consistency/claims — produced 55 findings; all load-bearing ones cross-validated against primary sources. Round 2's core fixes: (1) the M4 grading pipeline now has an explicit tiered oracle (it previously promised solver grading of multiway hands no solver can provide), (2) M2's answer keys are computed **in-house** (GTO Wizard's ToS bars scripted extraction, and it has no API — verified), (3) contracts, storage, and tech-stack decisions are pinned (§3b), (4) every milestone exit is now machine-checkable, (5) factual corrections (GTO Wizard 2026 pricing, ReBeL hardware attribution, WSOP event mix, PokerKit attribution). Rev 1 (original commercial-product framing) preserved at `poker-trainer-build-plan.orig.md`.

---

## 0. Framing: two goals, one project

**Goal A — Learning.** Implement the game theory (CFR family) and the cutting-edge RL (depth-limited search + learned value functions, best-response exploiters) with my own hands, at a scale where I can verify correctness.

**Goal B — Training.** Become a measurably +EV live full-ring MTT player. The bracelet is the aspiration; the *trainable* quantity is decision quality (EV loss per decision vs. best-available strategy). Variance decides the rest (§5.4).

**Explicit non-goal (v1): a commercial product.** No subscriptions, no premium UI polish, no production serving, no "beat GTO Wizard" positioning. The incumbent already ships styled exploitable opponents ("Profiles," Oct 2025) and neural real-time solving (GTO Wizard AI: ~6s to 0.22% exploitability on 2 cores/8GB; beat Slumbot by ~19 bb/100), so rev 1's "differentiators" are incumbent parity anyway. If commercial ambitions return later, that's a separate plan; the genuinely open niche is population-calibrated opponents (§6.2), which is parked here for data-legality reasons.

**The governing tradeoff:** every month spent building is a month not training. So — **buy the training, build the learning.** Training starts in week 1 on bought tools; the from-scratch builds are sequenced for what they teach.

---

## 1. Assumptions

**Executor.** Solo: me. Strong engineer, *new to game theory and RL* — learning them is a goal, so every estimate carries a learning tax.

**Hardware.** 64GB M-series MacBook Pro. Sufficient for: all development, single postflop solves (largest bet-grid/stack-depth trees may need tree reduction), MLX/Metal or PyTorch-MPS inference, toy-scale training. Cloud only for small bounded batch jobs.

**Budget.** Working ceiling **~$2k/year**, line-itemed in §8. The two product-scale jobs rev 1 implied (GTO-Wizard-scale presolve library: reviewed arithmetic $3k–$108k; full ReBeL-scale reproduction: six figures) stay cut.

**Games in scope.** NLHE. **Live full-ring MTT play with ICM is the primary training target.** Scope honesty (corrected in round 2): WSOP awards ~100 live bracelets plus ~30 online bracelets a year; roughly half the live events are non-hold'em, and of the live NLHE events ~8 are short-handed — so full-ring live NLHE is on the order of ~24 events/year. It is still the right primary target: it's the largest single live category, includes the marquee events (Main Event: 9,735 entries in 2025, ~15% cashed), and its skills transfer to the 6-max and online NLHE events. Non-NLHE games are out of scope and that consciously narrows bracelet coverage.

**Interim tooling (buy, don't build).** GTO Wizard **Premium** (~$950/yr at 2026 pricing) — the tier that includes ICM/bounty solutions and the trainer; Elite (~$1,670/yr) adds Profiles/nodelocking, which this plan doesn't need (we build our own bots). Note the 2026 restructure: Starter (~$470/yr) excludes ICM entirely. TexasSolver (open source, CPU build runs on the Mac; active development moved to the CUDA `TexasSolverGPU` fork, which doesn't) and/or PioSOLVER cover custom local solves. **Training starts in week 1.**

**Data & legal.**
- Trainer/RTA line unchanged: solver feedback lives only inside this sandbox; nothing overlays a real-money client.
- No scraped or purchased hand-history datasets (site ToS). Population modeling uses only aggregate/public stats and my own hands.
- **Solver-solution reuse (added in round 2, verified):** GTO Wizard has **no API and no bulk export** (ranges export is one-node-at-a-time manual copy), and its ToS bans automated requests/scripts (Art. 7.7) and third-party reuse of its ranges/trees/charts (Art. 7.5). Therefore: GTO Wizard is used **only inside its own apps** for study and practice; my own drill harness is fed **exclusively by self-generated artifacts** — my chart engine (M1.5) and my own TexasSolver/Pio/M3 solves, which I own. Nobody builds a scraper.

**Timeline.** No external deadline. Milestones are dependency-ordered (§8 includes the DAG) with machine-checkable exits, not calendar promises. (Solo-pacing reality check retained from round 1: rev 1's week-1–4 scope — engine + evaluator + toy-game CFR — is a ~6–10 week block for a CFR newcomer.)

**Definitions.**
- **Solver accuracy bar** (strategy quality): exploitability ≤0.5% of pot for NLHE subgames; NashConv ≤ 1e-3 chips (OpenSpiel units) for Kuhn/Leduc. Toy-game equilibria are non-unique, so exits compare *exploitability*, never strategy tables.
- **Decision ε** (scoring rule — distinct metric from the bar above): an action is *correct* if its EV loss ≤ ε (default 0.5% of pot, ≥0.1bb floor); in mixed spots any action with solver frequency ≥5% is acceptable, EV loss reported alongside.
- **"Instant"** = cache/chart lookup (<100ms). Live local solves taking seconds–minutes are acceptable; nothing user-facing blocks on a live solve.
- **SpotKey** = `(formation, stack_bucket, board_bucket)` — the library/grading lookup key. *Formation* = preflop position/action configuration (e.g., BTN-open vs BB-call). *Board bucket* = suit-isomorphism class + an 8-way texture label (paired? × monotone/two-tone/rainbow × connected/dry, collapsed to 8 classes). The `hand → SpotKey` resolver is a shared library function (§3b).
- **25-flop benchmark fixture**: a checked-in file (`benchmarks/flops25.json`) of 25 flops stratified across the 8 texture classes, each with fixed ranges (BTN-open vs BB-call, 40bb), stacks, and the 33/75/125%+jam grid. Referenced by M3 and anywhere "benchmark spots" appears.

---

## 2. Buy vs. build

**Buy / use as-is:** GTO Wizard Premium (study + practice inside their apps only — §1 legal), TexasSolver CPU / PioSOLVER (custom solves I own), OpenSpiel (reference CFR + toy-game ground truth), PokerKit (hand-mechanics correctness oracle — single-table only; it models no tournament state), phevaluator/OMPEval-class evaluator, Slumbot (HU benchmark; public HTTP API).

**Build, because it teaches (A) or because nothing ships it (B):**
1. **CFR family from scratch** (Kuhn → Leduc → HU NLHE subgames) + the **exploitability/best-response utility** that scores every solver exit (§3b). Borrow only the evaluator.
2. **Chart engine (M1.5)** — Malmuth-Harville ICM calculator + a small jam/fold equilibrium solver (fictitious play or CFR on the two-action game; a few hundred lines). Produces the push/fold, resteal, and ICM answer keys that M2 drills and M4 grading consume. In-house because it's cheap, legal (no vendor data), and is itself a Goal-A exercise.
3. **Toy-scale ReBeL / Student of Games** (Leduc → tiny-HUNL spike). No open-source poker ReBeL exists (Liar's Dice only) — from-paper reimplementation, scoped as a research spike with a go/no-go. Corrected hardware context: ReBeL's poker run used **a single machine for training** and up to ~1,024 GPUs (720 V100s in the poker experiment) **for data generation** — data generation is the real cost, which is why full scale stays out of scope.
4. **MTT drill harness** (§5.1) with a local-web UI — the daily training surface.
5. **Own-hand-history leak pipeline** (§5.3) — tiered grading, defined below.
6. *(Optional)* Styled bots (§6.1) — for learning value and as M6's fixed opponent.

---

## 3. Architecture

**One local app, one process, clean internal modules:** engine, solvers, chart engine, drill engine, HH importer/grader, value-net experiments, UI. Services only if a hosted product ever becomes a goal.

**Storage (pinned in round 2):** **SQLite** for app state — tables `imported_hands`, `gradings`, `drill_attempts`, `sr_state` (SM-2/Leitner), `solution_index`. Solve outputs live on disk (one file per solve, keyed by SpotKey) with the index in SQLite. **Derived metrics are queries, not tables** — leak rankings, EV-loss trends, and skill-gate stats are computed views over `gradings`/`drill_attempts`; nothing speculative is persisted. *As built (rev 3.3): two more tables ship, each backing a behaviour this plan promises — `failed_hands` (§8 M4's isolation) and `batch_queue` (§5.3's overnight batch). Both trace to concrete promised behaviour; neither is speculative persistence.*

**UI (pinned for M2):** local web view (table + range grid rendering beat a TUI for drills); functional only, no animation polish. The UI↔drill-engine contract is one loop: `next_spot() → Spot`, `submit_action(action) → Score` (per the §1 decision-ε rule). *As built (rev 3.2): the table view, drill loop, feedback (incl. ICM-$ units and prize ladder), and leak report shipped; the range grid is deferred to the post-review backlog — current drills are single-decision spots where the grid adds no answer-relevant information, and it becomes worth building alongside the deferred resteal/postflop drills whose answers are range-shaped. A deferral, not an oversight.* *Built (rev 3.4): `/api/drill/range` serves the 13×13 grid straight from the population's own Solution objects (no second source to drift), revealed after answering.*

### 3b. Contracts to pin before implementation (the parallelization surface)

1. **`GameState` / `Spot`** — seats, stacks (chips, with BB conversion), positions, action history, board, pot(s); plus **`TournamentContext`** = effective-BB, payout vector, players remaining. This is the shared type consumed by engine, solvers, drill engine, and HH grader. *Deliberately minimal:* no blind-clock simulation, no multi-table state — no milestone consumes them; add only when one does.
2. **`Solution`** — normalized solver output: per-infoset `{action → (ev, frequency)}` + range context. Every source (OpenSpiel, M3 solver, chart engine, manually seeded Pio-format text) adapts into this one type; scoring and grading depend only on it. *As built (rev 3.3): the `chart` and `subgame_solver` adapters ship; the `cfr` and `manual_seed` source values are declared in the contract but no adapter produces them — deferred until a consumer exists, per the same rule that scoped `GameState` minimal.*
3. **`hand → SpotKey` resolver** — §1's key schema, implemented once, used by library lookup and HH grading.
4. **Grading-tier contract** (§5.3): every decision routes to exactly one of chart / solver / best-available.
5. **`exploitability(strategy) / best_response(strategy)` utility** — built alongside M1 (CFR is unverifiable without it), reused by M1/M3/M5/M6/M7 exits. Distinct from the M6 *RL* exploiter (which attacks bots that aren't queryable analytically).
6. **Batched env API** (M6): `step(actions: [n]) → (obs: [n], rewards: [n], dones: [n])`, fixed observation/action encodings, exposed to Python via **pyo3**.

**Language decisions (pinned):** engine and everything through M5 in **Python** (correctness first, learning tax respected); Rust enters twice, deliberately — the solver hot loop inside M3 (after the Python version is correct) and the M6 vectorized env (an explicit port sharing M0's differential test suite). Value nets: **PyTorch + MPS** (ecosystem maturity); MLX only as an optional Metal-native excursion. **As-built deviation (rev 3.1):** both Rust items shipped as numpy-vectorized Python behind the same API contracts — no Rust toolchain on the build machine. The port seam is preserved (pinned contracts + the shared differential/BR test suites run against the API, not the implementation), so the Rust port remains a drop-in follow-up, not a rewrite.

---

## 4. Solver core (technical content unchanged from round 1 review; corrections applied)

- **Postflop subgame solver**: CFR+/DCFR over board/ranges/stacks/bet-grid. DCFR is a strong default; PCFR+/DDCFR/hyperparameter-schedule CFR are the 2024+ frontier if wanted later.
- **Borrowed evaluator** (tens of millions of evals/sec is real for lookup evaluators), **card isomorphism**, bet grid 33/75/125% + jam, full 1326-combo ranges on the flop.
- **Validation backbone**: differential-test the engine vs PokerKit on millions of scripted hands (hand mechanics); validate CFR vs OpenSpiel toy-game equilibria to the §1 bar before trusting any NLHE number. **PokerKit covers hand mechanics only** — the tournament layer (ICM math, payout ladders, blind levels) is tested separately against an independent ICM reference plus property tests (see M0/M1.5).
- **Multiway**: theory section stands — >2-player Nash has no meaningful optimality guarantees; Pluribus is empirical; no efficient general algorithm exists. Consequence for this plan: multiway content is **best-available, always labeled as such**, and the grading pipeline (§5.3) never pretends otherwise.
- **ICM**: enters at M1.5 (chart engine), not as a late-phase layer. Tournament context lives in `TournamentContext` (§3b) — the minimal fields drills actually read.
- **Real-time**: personal-scale answer — cached solves of spots I actually study + patient live solves. No <1s requirement.

---

## 5. Training program (Goal B)

### 5.1 MTT fundamentals first
Push/fold Nash (**≤20bb**, drilled across 5–20bb × positions), resteal, bubble and pay-jump ICM, final-table ICM, ante-adjusted full-ring open/defend ranges. **Answer keys come from my own chart engine (M1.5)** — computed, verifiable, ToS-clean. GTO Wizard is a manual cross-check during study, never a data source.

**Syllabus status (rev 3.2 — as-built honesty table; the round-3 review found the gap between this list and the shipped drill population was recorded nowhere):**
| item | status |
|---|---|
| Push/fold 5–20bb × SB/BB (ante-bucketed per §5.3) | **built** |
| Bubble ICM (4-player fixture set) | **built** |
| Resteal drills | **built (rev 3.4)** — re-jam/fold vs the priced open, only where the raise occurs at equilibrium |
| Ante-adjusted full-ring open/defend ranges | **built (rev 3.4)** — ring + open chain charts, full depth×ante grid, table sizes 2–9 |
| Final-table / pay-jump ladder variation | **built (rev 3.4)** — FT3/FT5 fixtures (`.icm.ft3`/`.icm.ft5`), bubble keys byte-identical |
| Postflop tier-2 drills from own solves | **built (rev 3.4)** — river drills from certified in-house solves (16 boards, gap ≤ 0.01bb, re-certified each run); flop/turn stay deferred behind the batch drain (a half-converged flop key would be fabricated) |
| Population-heuristic tier-3 flags as training signal | wired in wave 3 (flags persisted + reported) |
Deferrals are scope calls, not oversights — recorded here so the syllabus can be extended deliberately.

### 5.2 Multiway + postflop study
Full-ring single-raised and 3-bet pots, drilled from my own TexasSolver/M3 solves plus **population heuristics** — concretely: a frequency table (VPIP / PFR / 3-bet / c-bet / fold-vs-c-bet by position and street) sourced from public aggregate stats and my own imported hands, stored as a versioned config file, consumed by the drill generator (as priors on villain ranges) and by §6 bots (as target frequencies). Candidate-action count per drill is spot-dependent (the solver's meaningfully distinct options), not a fixed 3.

**As-built status (rev 3.3 — this section's honesty table, same treatment §5.1 got):**

| §5.2 item | Status |
|---|---|
| Population table, versioned config | shipped — `population/default.toml`, own observations only |
| Table schema | **deviates:** normalized action-type distributions (check/bet/call/fold/raise) keyed `formation\|street` — what tier-3 frequency-deviation flagging consumes — not the named VPIP/PFR/3-bet/c-bet/fold-vs-c-bet stats; documented in-file |
| Consumed by tier-3 grading | shipped (the §5.1 table's "wired wave 3" row) |
| Consumed by drill generator as range priors | **partially built (rev 3.4):** tier-3 multiway drills consume the table as their GRADING baseline (frequency flags, the §5.3 contract). Consumption as villain RANGE priors remains deferred — the table stores action-type distributions, not ranges (the schema deviation above), so there are no ranges in it to consume |
| Consumed by §6 bots as target frequencies | **deferred** — §6 bots ship with free knobs; table-derived targets land if/when population-styled bots are built |
| Postflop drills from these solves | deferred (recorded in §5.1) |

Deferrals are scope calls, not oversights.

### 5.3 My own hands — tiered grading (round-2 rewrite; this is the contract)
Import hand histories — **v1 sites: PokerStars (auto-saved local files) and GGPoker (self-service Pokercraft download; villain names anonymized, some fields unreliable — parser must tolerate)**. Other sites/converters and a manual-entry form for live hands are explicitly deferred. Parser output adapts into `GameState` (§3b).

Every decision routes to exactly one grading tier:
- **Tier 1 — chart-graded (exact, instant):** all preflop and ≤20bb jam/fold/resteal decisions, vs the M1.5 chart engine.
- **Tier 2 — solver-graded (accuracy bar applies):** heads-up (or HU-collapsed) postflop spots, vs the cached solution library (SpotKey lookup); **library misses are queued for overnight batch solves** — a session report is marked *partial* until its batch completes. Activates fully once M3 ships; before that, tier-2 spots fall to tier 3. *(Rev 3.2 honesty note: as built, tier-2 solves approximate both players' RANGES as uniform and stacks as symmetric — the accuracy bar applies to the solve given those inputs. The approximation is recorded in each Solution's provenance (`range_ctx`) and in the leak report; range modeling from the actual line is future work, not a silent assumption.)*
- **Tier 3 — best-available (labeled approximate):** genuine multiway postflop, graded against population-heuristic lines. **No EV-loss number is reported for tier 3** — only frequency-deviation flags — because per the plan's own theory section no trustworthy oracle exists there.

Leak taxonomy (pinned): categories keyed by the canonical 4-part key **formation × street × action-type × depth** (rev 3.1: depth added to match the build — a snapped bucket of 5/8/10/15/20bb for preflop jam/fold spots, `-` postflop; ICM variants take a `.icm` formation suffix; the key vocabulary is owned by one module, `drills/categories.py`, and shared verbatim by grading and spaced repetition — that shared key IS the leak→drill join), aggregated over tier-1/2 gradings only. "Top-5 leaks by EV-loss/100" is computed on the exact tiers; tier-3 deviations are listed separately. Spaced repetition (SM-2/Leitner over `sr_state`) resurfaces the worst categories as drills.

### 5.4 Variance realism
Trainable: EV per decision, range discipline, ICM execution. Not trainable: running hot through a 9,735-entry field. Target: **maximize bracelet equity per event entered**, measured by decision-quality trends over large samples, with a multi-year volume plan (online year-round; N bracelet events per summer). Note (round 2): MTT ROI needs tens of thousands of tournaments for statistical significance — ROI is a *directional* indicator here, never a pass/fail gate.

### 5.5 What this tool cannot train
Live tells, table talk, physical sizing reads, 12-hour stamina, tilt control, structure/clock management. Cover with live volume, a study group or coach, and mental-game work.

---

## 6. Styled & learned opponents

### 6.1 Parameterized perturbation bots
VPIP/aggression/bluff-frequency knobs over solver output. Convenience/learning build (GTO Wizard Profiles already ships archetypes commercially); also serves as M6's fixed opponent.

### 6.2 Population-priors bots
Calibrated only from the §5.2 heuristic table (aggregate/public stats + own hands). Site-scale calibration remains the open commercial niche — parked for data-legality reasons.

### 6.3 RL best-response exploiters (stretch, Goal A)
**Algorithm pinned:** exact/tabular best response at toy scale (the perturbation bot's strategy is queryable, so BR is computable directly and "measured exploitability" is exact); a deep-RL method (e.g., NFSP/PPO) only if scale ever genuinely demands it. Runs on the M6 vectorized env.

---

## 7. Learning ladder (Goal A)

- **L1.** CFR/CFR+/MCCFR in Python on Kuhn & Leduc + the exploitability/BR utility; verify vs OpenSpiel (~1–2 weeks each, realistically).
- **L2.** Python river-only HU solver → turn+river → full postflop subgame solver; port hot loop to Rust; benchmark vs TexasSolver (CPU) and the 25-flop fixture.
- **L3.** Depth-limited solving with a hand-rolled value net on Leduc (public belief states, CFR-D leaf values). PyTorch-MPS; Mac-tractable.
- **L4 (research spike, go/no-go).** **Tiny-HUNL, pinned:** 20bb effective, 2 bet sizes + jam, turn-and-river subgames only; training data = my own M3 solver's outputs sampled over i.i.d.-drawn subgames with flop iso-class recorded per row (rev 3.2: "SpotKey-stratified" dropped — the sampling was never stratified and the recorded runs would be invalidated by adding it now). Go/no-go: hard spend cap; **success = lower mean net-driven exploitability than the oracle-leaf depth-limited baseline on ≥20 held-out subgames** (rev 3.2: the baseline always existed as `mean_expl_oracle_bb`; an underived hardcoded 0.75bb constant briefly stood in for it in the go decision — removed, `SpikeVerdict.ratio` is the scale-free number). Full-scale reproduction stays out (withheld reference code; data-generation compute is the six-figure item — §2.3).
- **L5.** Best-response exploiter (§6.3); optional Slumbot ladder match for the bots.

(The ladder's L4/L5 order is inverted relative to milestones M6/M7 — they're independent tracks; either order works.)

Reading: Zinkevich CFR → CFR+ → DCFR → subgame re-solving/CFR-D → ReBeL → Student of Games → Pluribus.

---

## 8. Milestones — DAG, exits, costs

**Dependency graph** (∥ = parallel tracks):
```
M0 (engine, Python) ──→ M3 (HU solver) ──→ M4 tier-2 activation
   │                        ↑
   ├─→ M4 (HH pipeline: tiers 1+3 at launch)
   ├─→ M6 (Rust env port + exact-BR exploiter)
M1 (CFR + exploitability utility) ─→ M3, M5
M1.5 (chart engine) ─→ M2 (drill harness + UI)
M5 (toy ReBeL, Leduc) ─→ M7 (tiny-HUNL spike)
Training track: M1.5 → M2 → M4   ∥   Learning track: M1 → M3/M5 → M6/M7
```

| # | Deliverable | Exit criterion (machine-checkable) |
|---|---|---|
| M0 | Engine core, Python (`GameState`/`TournamentContext` per §3b) | Differential vs PokerKit: identical pot/stack/showdown on 1M random hands (hand mechanics); ICM/payout math matches an independent reference on a fixture set; blind-level/ante property tests pass. *(Rev 3.3, recording what the test docstring always said: one documented carve-out — an odd-chip split-placement class, a labelling difference among identically-ranked winners where totals and stacks still agree, is separately counted and tolerated under a ≤1% ceiling; everything else is exact.)* |
| M1 | CFR from scratch + exploitability/BR utility | NashConv ≤ 1e-3 chips on Kuhn & Leduc (own utility, cross-checked vs OpenSpiel) |
| M1.5 | Chart engine (ICM + jam/fold Nash) | ICM $ values match reference within 0.1%; jam/fold charts verified by **own exploitability utility** (~1e-6 on the two-action game; rev 3.3: measured ≤2.7e-6 across depths, CI bar tightened to 1e-5) plus ≥99% agreement with a hand-audited 33-entry reference set and a checked-in self-generated table as a drift guard. (Rev 3.1: "match published Nash charts" dropped — answer keys must be self-derived per §2/§5.1, so no vendor chart may serve as the oracle; correctness rests on exploitability, which is the stronger check.) |
| M2 | MTT drill harness + local-web UI | Drill sets (§5.1) load and score per the §1 decision-ε rule; **accuracy metric defined** = % of drilled spots where the chosen action clears the ε/frequency rule, over the fixed 5–20bb × position population; harness demonstrates ≥98% achievable on a chart-known reference agent |
| M3 | HU subgame solver (numpy hot loop; Rust port seam preserved — §3 deviation) | ≤0.5% pot exploitability (own BR utility, itself verified exactly vs brute force) on **river completions of the 25-flop fixture** — asserted in CI (fast: 1 fixture flop at the pinned bet grid; bench: 5 at a reduced grid); full-flop subgames assert convergence trend only, and the pinned-grid bar over all 25 flops is deferred behind the Rust hot-loop port (~1h/flop pure Python — recorded, not hidden); TexasSolver CPU agreement (root EV within 0.25% pot, per-action frequency L1 <10%) runs when the binary is installed. *(Rev 3.3 correction — the R4 audit caught this row overstating what ships: the export/consume pipeline is tested, but the two numeric bars are not computed anywhere; that gap was reviewed and accepted in round 1 — external binary, golden fixture deferred — so the earlier "never silently passed" phrasing promised an assertion that was deliberately deferred, not built. The bars remain the target if the comparison is ever promoted from pipeline-check to exit gate.)* |
| M4 | Own-HH pipeline (PS + GG parsers → tiered grading) | 100% of decisions in every **successfully imported** hand routed to a tier; hands failing parse/replay are isolated in `failed_hands` and surfaced in the session summary — never silently dropped, never partially graded (rev 3.1: isolation semantics made explicit); tiers 1 auto-graded instantly; tier-2 misses enqueued and batch-completed overnight *(rev 3.3: "overnight" is a deployment pattern, not a shipped scheduler — the user's own cron invokes `pokerlab-batch`; single-user tool, deliberately no scheduling daemon)*; leak report separates exact vs approximate tiers; top-5 leaks by EV-loss/100 computed on exact tiers |
| M5 | Toy ReBeL (Leduc) | Split criteria (rev 3.1, matching what CFR-D can actually guarantee): (a) depth-limited **safe re-solving with oracle leaf values** reaches NashConv ≤ 2e-3 (slow suite); (b) the trained value net is gated by root-value agreement \|v_net − v_full\| ≤ 2e-3 plus held-out RMSE (fast suite); (c) fully **net-driven** re-solving is measured and documented honestly — ≈0.42 NashConv ceiling from spurious round-1 equilibria (docs/notes/unsafe-resolving.md), a known property of unsafe decomposition, not a bug to fix |
| M6 | Vectorized env (numpy; Rust/pyo3 port seam preserved — §3 deviation) + exact-BR exploiter | Env passes M0's differential suite (via a single-env adapter); exact BR computed vs a fixed §6.1 bot; exploitability number reported |
| M7 | Tiny-HUNL value-net spike (L4) | Go/no-go honored (spend cap); go iff mean net-driven exploitability ≤ the oracle-leaf depth-limited baseline (× tolerance) on ≥20 held-out subgames, `SpikeVerdict.ratio` reported. As-built outcome: **NO-GO at ratio ≈ 17×** on two independent seeds, honestly recorded incl. the falsified loss-as-proxy attribution (docs/notes/tiny-hunl-spike.md) — the expected L4 result at local scale |

**Skill gates (measuring me, not the software):** EV-loss/100 trending down per leak category (exact tiers); ≥98% push/fold drill accuracy sustained; ITM% / deep-run rate tracked; online MTT ROI *directional only* (§5.4); **training-daily habit** tracked here as a KPI — deliberately not a software exit.

**Cost table (personal scope, annual):** local solves $0 · personal solve library ≤$100 cloud (or free overnight local) · toy training runs ≤$500 (spot GPUs) · GTO Wizard **Premium ~$950** → **total ~$1.5k/yr, under the ~$2k ceiling** (Elite would push ~$2.3k — not needed). Product-scale lines (presolve grid $3k–$108k, ReBeL-scale data generation, production serving) stay cut.

---

## 9. Risks

- **Building instead of training (meta-risk).** *De-risk:* GTO Wizard Premium from week 1; M1.5→M2 before any solver work if training pull is needed; training volume is a first-class KPI.
- **Grading over-claim.** The tier system exists so multiway spots are never silently graded as if a solver oracle existed. *De-risk:* tier labels are load-bearing UI, not fine print; EV-loss stats never mix tiers.
- **ToS compliance drift.** *De-risk:* the §1 rule is bright-line — vendor solutions never enter the harness; only self-generated artifacts do.
- **Toy-ReBeL spike fails.** Acceptable; L3 is the committed learning goal; nothing depends on M7.
- **Engine bugs poison downstream.** *De-risk:* M0 differential suite is reused by the M6 Rust port; solvers validated on toy games before NLHE.
- **Variance disappointment.** *De-risk:* §5.4 framing; decision quality and volume are the metrics, never tournament results.

---

## 10. Open source & references (corrections applied)

- **TexasSolver** — open-source HU postflop CFR+ solver; **CPU build runs on macOS** (active development continues in the CUDA-only `TexasSolverGPU` fork); M3's benchmark.
- **PokerKit** (MIT) — hand-mechanics correctness oracle; **single-table only, no tournament state**. From the group originally the UofT Computer Poker *Student* Research Group (de-affiliated and renamed, 2025).
- **OpenSpiel** — CFR variants + Kuhn/Leduc/universal_poker; equilibrium ground truth for M1.
- **RLCard / PettingZoo / Gymnasium** — RL env API scaffolding for M6.
- **phevaluator / OMPEval / 2+2** — evaluator hot loop.
- **Slumbot** — HU benchmark; public HTTP API (login/new_hand/act).
- **ReBeL** — paper (arXiv 2007.13544) + *Liar's-Dice-only* code release; poker implementation withheld. Poker run: single training machine; ~720 V100s (up to ~1,024 GPUs) for **data generation**.
- **Student of Games** (Science Advances 2023), **Pluribus** (Science 2019) — ladder reading.

---

### The one-line strategic summary

**Buy the training so it starts this week (inside the vendor's walls, ToS-clean); build the things that teach me game theory and RL; grade my real hands honestly — exact where an oracle exists, labeled-approximate where it doesn't — and measure me, not the software.** The bracelet stays what it is: variance riding on the best decision quality I can train.
