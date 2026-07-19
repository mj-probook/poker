# Build Plan: Personal Poker Lab — Learn Game Theory + RL, Train for a WSOP Bracelet

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

**Storage (pinned in round 2):** **SQLite** for app state — tables `imported_hands`, `gradings`, `drill_attempts`, `sr_state` (SM-2/Leitner), `solution_index`. Solve outputs live on disk (one file per solve, keyed by SpotKey) with the index in SQLite. **Derived metrics are queries, not tables** — leak rankings, EV-loss trends, and skill-gate stats are computed views over `gradings`/`drill_attempts`; nothing speculative is persisted.

**UI (pinned for M2):** local web view (table + range grid rendering beat a TUI for drills); functional only, no animation polish. The UI↔drill-engine contract is one loop: `next_spot() → Spot`, `submit_action(action) → Score` (per the §1 decision-ε rule).

### 3b. Contracts to pin before implementation (the parallelization surface)

1. **`GameState` / `Spot`** — seats, stacks (chips, with BB conversion), positions, action history, board, pot(s); plus **`TournamentContext`** = effective-BB, payout vector, players remaining. This is the shared type consumed by engine, solvers, drill engine, and HH grader. *Deliberately minimal:* no blind-clock simulation, no multi-table state — no milestone consumes them; add only when one does.
2. **`Solution`** — normalized solver output: per-infoset `{action → (ev, frequency)}` + range context. Every source (OpenSpiel, M3 solver, chart engine, manually seeded Pio-format text) adapts into this one type; scoring and grading depend only on it.
3. **`hand → SpotKey` resolver** — §1's key schema, implemented once, used by library lookup and HH grading.
4. **Grading-tier contract** (§5.3): every decision routes to exactly one of chart / solver / best-available.
5. **`exploitability(strategy) / best_response(strategy)` utility** — built alongside M1 (CFR is unverifiable without it), reused by M1/M3/M5/M6/M7 exits. Distinct from the M6 *RL* exploiter (which attacks bots that aren't queryable analytically).
6. **Batched env API** (M6): `step(actions: [n]) → (obs: [n], rewards: [n], dones: [n])`, fixed observation/action encodings, exposed to Python via **pyo3**.

**Language decisions (pinned):** engine and everything through M5 in **Python** (correctness first, learning tax respected); Rust enters twice, deliberately — the solver hot loop inside M3 (after the Python version is correct) and the M6 vectorized env (an explicit port sharing M0's differential test suite). Value nets: **PyTorch + MPS** (ecosystem maturity); MLX only as an optional Metal-native excursion.

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

### 5.2 Multiway + postflop study
Full-ring single-raised and 3-bet pots, drilled from my own TexasSolver/M3 solves plus **population heuristics** — concretely: a frequency table (VPIP / PFR / 3-bet / c-bet / fold-vs-c-bet by position and street) sourced from public aggregate stats and my own imported hands, stored as a versioned config file, consumed by the drill generator (as priors on villain ranges) and by §6 bots (as target frequencies). Candidate-action count per drill is spot-dependent (the solver's meaningfully distinct options), not a fixed 3.

### 5.3 My own hands — tiered grading (round-2 rewrite; this is the contract)
Import hand histories — **v1 sites: PokerStars (auto-saved local files) and GGPoker (self-service Pokercraft download; villain names anonymized, some fields unreliable — parser must tolerate)**. Other sites/converters and a manual-entry form for live hands are explicitly deferred. Parser output adapts into `GameState` (§3b).

Every decision routes to exactly one grading tier:
- **Tier 1 — chart-graded (exact, instant):** all preflop and ≤20bb jam/fold/resteal decisions, vs the M1.5 chart engine.
- **Tier 2 — solver-graded (accuracy bar applies):** heads-up (or HU-collapsed) postflop spots, vs the cached solution library (SpotKey lookup); **library misses are queued for overnight batch solves** — a session report is marked *partial* until its batch completes. Activates fully once M3 ships; before that, tier-2 spots fall to tier 3.
- **Tier 3 — best-available (labeled approximate):** genuine multiway postflop, graded against population-heuristic lines. **No EV-loss number is reported for tier 3** — only frequency-deviation flags — because per the plan's own theory section no trustworthy oracle exists there.

Leak taxonomy (pinned): categories keyed by **formation × street × action-type**, aggregated over tier-1/2 gradings only. "Top-5 leaks by EV-loss/100" is computed on the exact tiers; tier-3 deviations are listed separately. Spaced repetition (SM-2/Leitner over `sr_state`) resurfaces the worst categories as drills.

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
- **L4 (research spike, go/no-go).** **Tiny-HUNL, pinned:** 20bb effective, 2 bet sizes + jam, turn-and-river subgames only; training data = my own M3 solver's outputs sampled over SpotKey-stratified subgames (~10⁴–10⁵ rows, Parquet of belief-state features → values). Go/no-go: hard spend cap; **success = lower mean exploitability than the L3-style depth-limited baseline on ≥20 held-out subgames.** Full-scale reproduction stays out (withheld reference code; data-generation compute is the six-figure item — §2.3).
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
| M0 | Engine core, Python (`GameState`/`TournamentContext` per §3b) | Differential vs PokerKit: identical pot/stack/showdown on 1M random hands (hand mechanics); ICM/payout math matches an independent reference on a fixture set; blind-level/ante property tests pass |
| M1 | CFR from scratch + exploitability/BR utility | NashConv ≤ 1e-3 chips on Kuhn & Leduc (own utility, cross-checked vs OpenSpiel) |
| M1.5 | Chart engine (ICM + jam/fold Nash) | ICM $ values match reference within 0.1%; jam/fold ranges match published Nash charts on ≥99% of hands at 5–20bb |
| M2 | MTT drill harness + local-web UI | Drill sets (§5.1) load and score per the §1 decision-ε rule; **accuracy metric defined** = % of drilled spots where the chosen action clears the ε/frequency rule, over the fixed 5–20bb × position population; harness demonstrates ≥98% achievable on a chart-known reference agent |
| M3 | HU subgame solver (Rust hot loop) | ≤0.5% pot exploitability on the 25-flop fixture (own BR utility); agreement vs TexasSolver CPU: root EV within 0.25% pot and per-action frequency L1 distance <10% at fixture root nodes |
| M4 | Own-HH pipeline (PS + GG parsers → tiered grading) | 100% of decisions in a test session routed to a tier; tiers 1 auto-graded instantly; tier-2 misses enqueued and batch-completed overnight; leak report separates exact vs approximate tiers; top-5 leaks by EV-loss/100 computed on exact tiers |
| M5 | Toy ReBeL (Leduc) | Depth-limited + value net reaches NashConv ≤ 2e-3 on Leduc (≤2× the M1 bar) |
| M6 | Rust vectorized env (pyo3, shared M0 test suite) + exact-BR exploiter | Env passes M0's differential suite; exact BR computed vs a fixed §6.1 bot; exploitability number reported |
| M7 | Tiny-HUNL value-net spike (L4) | Go/no-go honored (spend cap); if go: beats depth-limited baseline on ≥20 held-out subgames (mean exploitability) |

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
