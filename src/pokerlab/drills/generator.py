"""Drill generator (Slice E; plan §5.1 MTT fundamentals).

Turns the in-house chart engine (Slice C) into a drill population:

  * jam/fold drills across {5,8,10,15,20}bb × {SB open-jam, BB call-vs-jam},
    169 hand classes each — the ≤20bb push/fold syllabus of plan §5.1;
  * an ICM bubble set from a fixture `TournamentContext` (4 players, 3 paid),
    solved with `charts.solve_jamfold_icm`.

Each `Drill` carries the normalized `Solution` scoring reads, the pot context
for the decision-ε rule, and a category `leak_key` for persistence + SM-2. The
answer keys are 100% self-generated (chart engine only) — nothing vendor-derived
(CLAUDE.md hard rule).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from pokerlab.charts import hands, jamfold_range
from pokerlab.charts.equity import load_equity_matrix
from pokerlab.charts.jamfold import icm_model, joint_prior, solve_jamfold_icm
from pokerlab.charts.openraise import (OPEN_FORMATIONS, open_ctx,
                                       open_solution)
from pokerlab.charts.ring import (RING_JAMMERS, RING_ORDER,
                                  ring_defense_range, ring_range)
from pokerlab.drills.categories import (ANTES, jamfold_category,
                                        open_category, resteal_category,
                                        ring_category)
from pokerlab.types import TIER_CHART, Solution, TournamentContext

DEPTHS: tuple[int, ...] = (5, 8, 10, 15, 20)

# Bubble fixture: 4 players, top 3 paid, 10bb effective (matches Slice C's
# ICM directional test so the two stay consistent).
BUBBLE = TournamentContext(
    payouts=(500, 300, 200),
    players_remaining=4,
    stacks_all=(1000, 1000, 1000, 1000),
    bb=100,
    ante=0,
)

_ACTIONS: dict[str, tuple[str, ...]] = {"SB": ("jam", "fold"), "BB": ("call", "fold")}

# Final-table fixtures (plan-deferred drill kinds, built 2026-07-29). FT3 is
# pure LADDER pressure — everyone is paid, the money at stake is the jumps —
# with asymmetric stacks (short SB shoving into the chip leader's BB at 8bb
# effective). FT5 is the final-table money bubble: 5 left, 3 paid.
FT3 = TournamentContext(
    payouts=(500, 300, 200),
    players_remaining=3,
    stacks_all=(1200, 1000, 800),
    bb=100,
    ante=0,
)
FT5 = TournamentContext(
    payouts=(500, 300, 200),
    players_remaining=5,
    stacks_all=(1000, 1000, 1000, 1000, 1000),
    bb=125,
    ante=0,
)


@dataclass(frozen=True)
class Drill:
    """A single scored spot: chart `Solution` + pot context + persistence keys."""

    drill_id: str                 # unique: "<leak_key>:<hand>"
    kind: str                     # "jamfold" | "icm"
    position: str                 # "SB" | "BB"
    depth_bb: float
    hand_label: str               # 169-class label, e.g. "AKs"
    solution: Solution
    pot_bb: float                 # pot the decision plays for (decision-ε input)
    leak_key: str                 # canonical category (drills.categories)
    legal_actions: tuple[str, ...]
    description: str
    tournament: TournamentContext | None = None
    # What one big blind is worth in this drill's payoff currency, i.e. the
    # factor that puts the decision-ε in the same units as the solution's EVs
    # (round-3 finding [P7']). 1.0 for chip-EV drills, whose EVs are already
    # bb; the average chip's dollar value for ICM drills, whose EVs are
    # $-deltas. Carried on the Drill so the scoring call site cannot forget it.
    bb_value: float = 1.0
    # DISTRACTOR actions: submittable but outside the solved game, so they
    # grade as framework deviations with ev_loss None — the chart never
    # priced them, and pretending it did (any number, including 0) would be
    # the fabricated-claim class this project bans. First-in spots carry
    # limp/raise distractors; spots facing an all-in carry NONE, because
    # poker itself allows only call or fold there — a false button would be
    # a rendered false fact, not a pedagogical trick.
    off_tree_actions: tuple[str, ...] = ()
    # Server-authored sentence for the page when the action set is complete
    # at two (the page renders server words, never its own claims).
    action_note: str = ""
    # The position whose aggression the hero is facing ("" when hero is first
    # to act). A formation fact the scene renders — "SB is all-in for 10bb" —
    # and the RTA framing depends on.
    versus: str = ""
    # WHAT that position did: "all-in" or a raise label ("raise 2.2bb").
    # Empty iff `versus` is empty. The scene draws the difference (all-in
    # badge + full stack pushed vs a raise badge + the raise amount).
    versus_action: str = ""
    # Seat list in action order when seat attribution is a formation fact
    # (HU + 9-max ring). Empty for ICM multiway, whose payload deliberately
    # does not attribute stacks to seats.
    table: tuple[str, ...] = ()
    # Grading tier (types.TIER_*). Chart drills are tier 1; postflop drills
    # graded by the subgame solver are tier 2 — the page shows the tier's
    # honest label, so it must be a Drill fact, not a page guess.
    tier: int = TIER_CHART
    # Community cards as concrete strings ("As", ...) for postflop drills;
    # empty preflop. When the board is real, the hero's cards must be too
    # (suits ARE the strategy on a board), hence hero_cards; preflop drills
    # keep the 169-class label and synthetic display suits.
    board: tuple[str, ...] = ()
    hero_cards: tuple[str, ...] = ()
    # Tier-3 only: the population table's key for this spot ("6max:BB") —
    # the ONLY grading reference a multiway drill has (frequency flags,
    # never EV). Empty for tiers 1/2, whose Solutions carry priced actions.
    population_key: str = ""
    # Tier-3 only: clearly-labeled ADVISORY analysis (HU-collapsed solve
    # numbers, MC equity, and the caveat sentence the page must render with
    # them). Never consulted by grading — study aid, not answer key.
    advisory: dict | None = None


# SB open distractors: plausible at the table, unpriced by the jam/fold chart.
_OFF_TREE_SB = ("limp", "raise 2.2bb", "raise 3bb")
_BB_NOTE = ("Facing an all-in, raise does not exist: the only actions are "
            "call and fold.")
# Open-game notes. Limp is the ONE table action the open game still cannot
# price (a limped pot reaches postflop), so it stays a distractor there; a
# flat-call of a raise is unpriced for the same reason on resteal spots.
_OPEN_NOTE = ("Raises here are priced by the open-game chart. Limping is not "
              "— a limped pot plays postflop, which this chart does not "
              "model.")
_RESTEAL_NOTE = ("Flat-calling the raise leads to postflop play this chart "
                 "does not price; the solved framework is re-jam or fold.")
# A raise must actually OCCUR at equilibrium for the defend-vs-raise node to
# be solved: an unreached CFR infoset carries an arbitrary strategy, and
# serving drills from it would grade the user against noise. One combo is the
# floor for "this raise is a real part of the solution".
_MIN_RAISE_COMBOS = 1.0


def _describe(pos: str, depth: float, hand: str, kind: str,
              tc: TournamentContext | None, ante_bb: float = 0.0) -> str:
    d = int(depth)
    # The ante is part of the QUESTION, not decoration: the three ante variants
    # of one spot have genuinely different answer keys (15 chart flips at
    # 0.125bb/player, 25 at 0.25 — round-3 finding [E49]), so a prompt that
    # omitted it would ask the user to guess which chart is being tested.
    ante = f", {ante_bb:g}bb ante" if ante_bb else ""
    # The SB prompt must NOT enumerate the priced pair ("open-jam or fold?"):
    # the button row now carries off-tree distractors, and a prompt naming the
    # real options would identify them. The BB prompt keeps "call or fold?" —
    # facing an all-in that pair is poker-complete, which is also why BB spots
    # carry no distractors.
    # The HU drills name their table ("heads-up") now that a 9-max SB
    # formation exists — "SB 10bb" alone no longer says which chart answers.
    # The ICM prompts keep their own table context (the bracket prefix).
    hu = ", heads-up" if kind == "jamfold" else ""
    if pos == "SB":
        base = f"SB {d}bb{ante}{hu}, {hand}: your action?"
    else:
        base = f"BB {d}bb{ante}{hu} facing an SB all-in, {hand}: call or fold?"
    if kind == "icm" and tc is not None:
        # derived from the ladder, not from a label: fewer paid than left is
        # a bubble; everyone paid is final-table ladder pressure
        tag = ("Bubble ICM" if len(tc.payouts) < tc.players_remaining
               else "Final-table ICM")
        base = (f"[{tag} · {tc.players_remaining} left, "
                f"{len(tc.payouts)} paid] " + base)
    return base


def jamfold_drills(depths: tuple[int, ...] = DEPTHS) -> list[Drill]:
    """Chip-EV push/fold drills over depths × antes × {SB, BB} × 169 classes.

    Every ante bucket gets its own drills (round-3 finding [E49]): the grader
    keys a real hand's category at its snapped ante, so a category the
    generator never emits is a leak the training loop can detect but not
    train. The ante is solved at the SAME bucket the key names — both sides go
    through `categories.snap_ante`.
    """
    out: list[Drill] = []
    for pos in ("SB", "BB"):
        for d in depths:
            for ante in ANTES:
                rng = jamfold_range(pos, float(d), ante)
                leak_key = jamfold_category(pos, float(d), ante_bb=ante)
                pot = 2.0 * float(d)
                for hand in hands.HAND_CLASSES:
                    out.append(Drill(
                        drill_id=f"{leak_key}:{hand}", kind="jamfold",
                        position=pos, depth_bb=float(d), hand_label=hand,
                        solution=rng[hand], pot_bb=pot, leak_key=leak_key,
                        legal_actions=_ACTIONS[pos],
                        description=_describe(pos, d, hand, "jamfold", None, ante),
                        off_tree_actions=_OFF_TREE_SB if pos == "SB" else (),
                        action_note="" if pos == "SB" else _BB_NOTE,
                        versus="" if pos == "SB" else "SB",
                        versus_action="" if pos == "SB" else "all-in",
                        table=("SB", "BB"),
                    ))
    return out


@lru_cache(maxsize=1)
def _ring_drills_cached() -> tuple[Drill, ...]:
    return tuple(_build_ring_drills())


def ring_drills() -> list[Drill]:
    return list(_ring_drills_cached())


def _build_ring_drills() -> list[Drill]:
    """First-in jam drills for every non-blind position, plus a defend drill
    for every (responder, jammer) pair — at EVERY table size the artifact
    ships (3–9 players since the table-size axis; charts/ring.py; the
    model's three restrictions ride in each Solution's range_ctx).

    Artifact-DRIVEN on purpose: the population is whatever the checked-in
    charts certify, never a live solve — a missing key means those drills do
    not exist yet, not a multi-minute import."""
    from pokerlab.charts.ring import load_ring_charts

    out: list[Drill] = []
    for sol in load_ring_charts().values():
        jammer, table = sol.jammer, sol.table
        size = len(table)
        d, ante = sol.depth_bb, sol.ante
        ring_tag = "9-max" if size == 9 else f"{size}-max"
        a = f", {ante:g}bb ante" if ante else ""
        pot = 2.0 * d
        behind = table[table.index(jammer) + 1:]
        rng = ring_range(jammer, d, ante, table_size=size)
        leak = ring_category(jammer, d, ante_bb=ante, table_size=size)
        for hand in hands.HAND_CLASSES:
            out.append(Drill(
                drill_id=f"{leak}:{hand}", kind="ring",
                position=jammer, depth_bb=d, hand_label=hand,
                solution=rng[hand], pot_bb=pot, leak_key=leak,
                legal_actions=("jam", "fold"),
                description=(f"{jammer} {int(d)}bb{a}, {ring_tag} "
                             f"first-in, {hand}: your action?"),
                off_tree_actions=_OFF_TREE_SB,
                versus="", table=table,
            ))
        for resp in behind:
            rngd = ring_defense_range(resp, versus=jammer, depth_bb=d,
                                      ante=ante, table_size=size)
            leakd = ring_category(resp, d, versus=jammer, ante_bb=ante,
                                  table_size=size)
            for hand in hands.HAND_CLASSES:
                out.append(Drill(
                    drill_id=f"{leakd}:{hand}", kind="ring",
                    position=resp, depth_bb=d,
                    hand_label=hand, solution=rngd[hand],
                    pot_bb=pot, leak_key=leakd,
                    legal_actions=("call", "fold"),
                    description=(f"{resp} {int(d)}bb{a} facing a "
                                 f"{jammer} all-in ({ring_tag}), {hand}: "
                                 f"call or fold?"),
                    action_note=_BB_NOTE,
                    versus=jammer, versus_action="all-in",
                    table=table,
                ))
    return out


def _icm_solutions(tc: TournamentContext, sb_seat: int, bb_seat: int
                   ) -> tuple[dict[str, Solution], dict[str, Solution], float]:
    """{hand: Solution} for SB and BB under ICM-$ payoffs, plus effective depth.

    Frequencies come straight from `solve_jamfold_icm`. Per-hand EVs are the
    real ICM $-deltas, recomputed here from the *public* chart model (same
    reach-weighted normalization the chip solver uses) because the general-sum
    ICM solve does not expose them. ev_loss for ICM is therefore a $-delta.

    That $-delta is graded by the decision-ε rule converted into ICM-$ (see
    `scoring.epsilon`), NOT by frequency. This docstring previously claimed
    "ICM correctness is frequency-driven — the honest read for a general-sum
    spot", which described behavior the code never had (round-3 finding
    [P7']): the 5% mixed-spot hatch opened 0 times in 338 ICM drills, because
    this solve is essentially pure (max second-action frequency 0.0009). With
    both the frequency hatch and the unit-mismatched ε branch dead, grading was
    exact-argmax. These are exact solves from our own engine, so they deserve
    gradeable drills rather than tier-3 posture — the honesty obligation here
    is to LABEL the number as ICM-$, which the web layer now does.
    """
    stacks = list(tc.stacks_all)
    payouts = list(tc.payouts)
    sol = solve_jamfold_icm(tuple(stacks), sb_seat, bb_seat,
                            tuple(payouts), tc.bb, float(tc.ante))
    E = load_equity_matrix().equity_matrix
    P = joint_prior()
    w = P.sum(axis=1)
    model = icm_model(stacks, sb_seat, bb_seat, payouts, tc.bb, float(tc.ante), E)
    x, y = sol.sb_jam, sol.bb_call

    jam_ev = (P * (y[None, :] * model.call_sb
                   + (1.0 - y)[None, :] * model.bbfold_sb)).sum(axis=1) / w
    reach_bb = (P * x[:, None]).sum(axis=0)
    call_ev = (P * x[:, None] * model.call_bb).sum(axis=0) \
        / np.where(reach_bb > 0, reach_bb, 1.0)

    sb: dict[str, Solution] = {}
    bb: dict[str, Solution] = {}
    ctx = (f"chart:jamfold_icm:{tc.players_remaining}p:"
           f"{len(payouts)}paid:{sol.depth_bb}bb")
    for i, label in enumerate(hands.HAND_CLASSES):
        fx = float(x[i])
        sb[label] = Solution(
            actions={"jam": (float(jam_ev[i]), fx),
                     "fold": (float(model.sbfold_sb), 1.0 - fx)},
            range_ctx=ctx, source="chart")
        fy = float(y[i])
        bb[label] = Solution(
            actions={"call": (float(call_ev[i]), fy),
                     "fold": (float(model.bbfold_bb), 1.0 - fy)},
            range_ctx=ctx, source="chart")
    return sb, bb, float(sol.depth_bb)


def icm_drills(tournament: TournamentContext = BUBBLE, sb_seat: int = 0,
               bb_seat: int = 1, label: str = "") -> list[Drill]:
    """ICM push/fold drills from a fixture TournamentContext.

    `label` namespaces the categories (`.icm.ft3`) so fixtures never share an
    answer key; the empty label is the original bubble fixture, whose keys
    must stay byte-identical.
    """
    sb_sol, bb_sol, depth = _icm_solutions(tournament, sb_seat, bb_seat)
    pot = 2.0 * depth
    out: list[Drill] = []
    # ICM antes are NOT expanded into variants: unlike the chip-EV charts, the
    # ante here is a property of the fixture tournament, not a free axis, so
    # the key names the ante that context actually has (round-3 finding [E49]).
    icm_ante = float(tournament.ante) / float(tournament.bb) if tournament.bb else 0.0
    # Average chip value: the prize pool spread over every chip in play, in bb.
    # Converts the bb-denominated decision-ε into the ICM-$ the payoffs use
    # (round-3 finding [P7']); see scoring.epsilon for the derivation.
    total_bb = sum(tournament.stacks_all) / float(tournament.bb)
    bb_value = (float(sum(tournament.payouts)) / total_bb) if total_bb else 1.0
    for pos, sols in (("SB", sb_sol), ("BB", bb_sol)):
        leak_key = jamfold_category(pos, depth, icm=True,
                                    ante_bb=icm_ante, icm_label=label)
        for hand in hands.HAND_CLASSES:
            out.append(Drill(
                drill_id=f"{leak_key}:{hand}", kind="icm", position=pos,
                depth_bb=depth, hand_label=hand, solution=sols[hand],
                pot_bb=pot, leak_key=leak_key, legal_actions=_ACTIONS[pos],
                description=_describe(pos, depth, hand, "icm", tournament, icm_ante),
                tournament=tournament, bb_value=bb_value,
                off_tree_actions=_OFF_TREE_SB if pos == "SB" else (),
                action_note="" if pos == "SB" else _BB_NOTE,
                versus="" if pos == "SB" else "SB",
                versus_action="" if pos == "SB" else "all-in",
                # table deliberately stays empty: the ICM payload does not
                # attribute stacks to seats, so the scene keeps them anonymous
            ))
    return out


@lru_cache(maxsize=1)
def _open_drills_cached() -> tuple[Drill, ...]:
    return tuple(_build_open_drills())


def open_drills() -> list[Drill]:
    return list(_open_drills_cached())


def _build_open_drills() -> list[Drill]:
    """First-in OPEN drills: fold / raise 2.2bb / raise 3bb / jam, all priced
    (charts/openraise.py), for every 9-max position plus the HU SB — the
    answer to "the raise buttons should be real options". Limp remains the
    one distractor, with the note saying why.

    Also emits the RESTEAL drills that fall out of the same solve: re-jam or
    fold facing a named raise size, for every responder — but ONLY where that
    raise actually occurs at equilibrium (raise combo mass ≥ 1): an unreached
    defend node's CFR strategy is arbitrary, and a drill graded against noise
    would be a fabricated answer key wearing a real one's clothes.
    """
    from pokerlab.charts.openraise import load_open_charts, open_cache_key

    charts = load_open_charts()
    out: list[Drill] = []
    for formation, (table, opener) in OPEN_FORMATIONS.items():
        behind = table[table.index(opener) + 1:]
        hu = formation == "SBhu"
        ring_tag = ("heads-up" if hu
                    else "9-max" if len(table) == 9
                    else f"{len(table)}-max")
        for d in DEPTHS:
            for ante in ANTES:
                # artifact-driven, like ring_drills: a formation the shipped
                # charts do not certify has no drills, not a live solve
                if open_cache_key(formation, float(d), float(ante)) not in charts:
                    continue
                a = f", {ante:g}bb ante" if ante else ""
                pot = 2.0 * float(d)
                sol = open_solution(formation, float(d), ante)
                ctx = open_ctx(sol, formation)
                labels = ["jam"] + [f"raise {r:g}bb" for r in sol.sizes] + ["fold"]
                leak = open_category(formation, float(d), ante_bb=ante)
                for i, hand in enumerate(hands.HAND_CLASSES):
                    actions = {lab: (float(sol.open_ev[lab][i]),
                                     float(sol.open_freq[lab][i]))
                               for lab in labels}
                    out.append(Drill(
                        drill_id=f"{leak}:{hand}", kind="open",
                        position=opener, depth_bb=float(d), hand_label=hand,
                        solution=Solution(actions=actions, range_ctx=ctx,
                                          source="chart"),
                        pot_bb=pot, leak_key=leak,
                        legal_actions=tuple(labels),
                        description=(f"{opener} {int(d)}bb{a}, {ring_tag} "
                                     f"first-in, {hand}: your action?"),
                        off_tree_actions=("limp",), action_note=_OPEN_NOTE,
                        table=table,
                    ))
                for r in sol.sizes:
                    rlab = f"raise {r:g}bb"
                    mass = float(sum(
                        f * hands.combos(h) for f, h in
                        zip(sol.open_freq[rlab], hands.HAND_CLASSES)))
                    if mass < _MIN_RAISE_COMBOS:
                        continue
                    for q in behind:
                        y = sol.defend_freq[(q, rlab)]
                        ev = sol.defend_ev[(q, rlab)]
                        fev = sol.defend_fold_ev[(q, rlab)]
                        leakd = resteal_category(q, formation, float(r),
                                                 float(d), ante_bb=ante)
                        for i, hand in enumerate(hands.HAND_CLASSES):
                            actions = {
                                "jam": (float(ev[i]), float(y[i])),
                                "fold": (float(fev), 1.0 - float(y[i]))}
                            out.append(Drill(
                                drill_id=f"{leakd}:{hand}", kind="resteal",
                                position=q, depth_bb=float(d),
                                hand_label=hand,
                                solution=Solution(actions=actions,
                                                  range_ctx=ctx,
                                                  source="chart"),
                                pot_bb=pot, leak_key=leakd,
                                legal_actions=("jam", "fold"),
                                description=(
                                    f"{q} {int(d)}bb{a} facing a {opener} "
                                    f"raise to {r:g}bb ({ring_tag}), {hand}: "
                                    f"your action?"),
                                off_tree_actions=("call",),
                                action_note=_RESTEAL_NOTE,
                                versus=opener, versus_action=rlab,
                                table=table,
                            ))
    return out


def default_population() -> list[Drill]:
    """Cached per process: the population is deterministic and ~850k Drill
    objects — every web-test module (and every create_app) re-deriving it
    would cost seconds and a duplicate gigabyte per build. A fresh list is
    returned each call; the Drills themselves are shared and read-only."""
    return list(_population_cached())


@lru_cache(maxsize=1)
def _population_cached() -> tuple[Drill, ...]:
    return tuple(_build_population())


def _build_population() -> list[Drill]:
    """The full drill population served by the web app: HU jam/fold + ICM
    bubble + 9-max ring + the open game (priced raises) + resteals. One
    function on purpose — the app-wide honesty pins (every kind has an EV
    unit, every source a verb, the scheduler budget counts every category)
    sweep THIS list, so a population source that lived outside it would
    silently escape them."""
    # imported here, not at module top: river.py/multiway.py import Drill
    # from THIS module, so top-level imports would be circular
    from pokerlab.drills.multiway import multiway_drills
    from pokerlab.drills.river import river_drills

    return (jamfold_drills() + icm_drills()
            + icm_drills(FT3, sb_seat=2, bb_seat=0, label="ft3")
            + icm_drills(FT5, sb_seat=0, bb_seat=1, label="ft5")
            + ring_drills() + open_drills() + river_drills()
            + multiway_drills())


def sample_drills(drills: list[Drill], n: int, seed: int) -> list[Drill]:
    """Deterministic mixed sample of `n` drills (with replacement if n > pool)."""
    rng = random.Random(seed)
    if n <= len(drills):
        return rng.sample(drills, n)
    return [rng.choice(drills) for _ in range(n)]
