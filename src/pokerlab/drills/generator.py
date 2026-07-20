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

import numpy as np

from pokerlab.charts import hands, jamfold_range
from pokerlab.charts.equity import load_equity_matrix
from pokerlab.charts.jamfold import icm_model, joint_prior, solve_jamfold_icm
from pokerlab.drills.categories import ANTES, jamfold_category
from pokerlab.types import Solution, TournamentContext

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


def _describe(pos: str, depth: float, hand: str, kind: str,
              tc: TournamentContext | None, ante_bb: float = 0.0) -> str:
    d = int(depth)
    # The ante is part of the QUESTION, not decoration: the three ante variants
    # of one spot have genuinely different answer keys (15 chart flips at
    # 0.125bb/player, 25 at 0.25 — round-3 finding [E49]), so a prompt that
    # omitted it would ask the user to guess which chart is being tested.
    ante = f", {ante_bb:g}bb ante" if ante_bb else ""
    if pos == "SB":
        base = f"SB {d}bb{ante}, {hand}: open-jam or fold?"
    else:
        base = f"BB {d}bb{ante} facing an SB all-in, {hand}: call or fold?"
    if kind == "icm" and tc is not None:
        base = (f"[Bubble ICM · {tc.players_remaining} left, "
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
               bb_seat: int = 1) -> list[Drill]:
    """Bubble ICM push/fold drills from a fixture TournamentContext."""
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
        leak_key = jamfold_category(pos, depth, icm=True, ante_bb=icm_ante)
        for hand in hands.HAND_CLASSES:
            out.append(Drill(
                drill_id=f"{leak_key}:{hand}", kind="icm", position=pos,
                depth_bb=depth, hand_label=hand, solution=sols[hand],
                pot_bb=pot, leak_key=leak_key, legal_actions=_ACTIONS[pos],
                description=_describe(pos, depth, hand, "icm", tournament, icm_ante),
                tournament=tournament, bb_value=bb_value,
            ))
    return out


def default_population() -> list[Drill]:
    """The full drill population served by the web app: jam/fold + ICM bubble."""
    return jamfold_drills() + icm_drills()


def sample_drills(drills: list[Drill], n: int, seed: int) -> list[Drill]:
    """Deterministic mixed sample of `n` drills (with replacement if n > pool)."""
    rng = random.Random(seed)
    if n <= len(drills):
        return rng.sample(drills, n)
    return [rng.choice(drills) for _ in range(n)]
