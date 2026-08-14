"""Per-decision grading (Slice F, impl doc §3 / §5.3).

Tier dispatch, honest by construction:

  * tier 1 (chart)  — ≤20bb SB-jam / BB-call-vs-jam spots graded against
    `charts.jamfold_range` via the shared decision-ε rule (`drills.scoring`).
    ``ev_loss`` is populated (bb conceded to the best action).
  * tier 2 (solver) — HU postflop; graded only if a `Solution` is supplied
    (from the solution library). No solution -> marked pending (batch worker).
  * tier 3 (best-available) — genuine multiway postflop. ``ev_loss`` is ALWAYS
    None (grading-honesty hard rule); the only signal is a frequency-deviation
    flag versus the population table.

`Grading` mirrors the `store.gradings` row (hand_id is filled at persist time).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pokerlab.charts.jamfold import jamfold_range
from pokerlab.drills.categories import (
    jamfold_category,
    postflop_category,
    ring_category,
    snap_ante,
    snap_depth,
)
from pokerlab.drills.scoring import score
from pokerlab.hh.decisions import Decision, hand_label
from pokerlab.hh.population import Population
from pokerlab.types import (
    Solution,
    TIER_BEST_AVAILABLE,
    TIER_CHART,
    TIER_SOLVER,
)

JAMFOLD_MAX_BB = 20.0     # push/fold routing/grading ceiling (plan §5.3)
RARE_ACTION_FREQ = 0.10   # population freq below which an action is flagged


@dataclass
class Grading:
    decision_index: int
    tier: int
    chosen: str
    best: str | None
    ev_loss: float | None      # None for tier 3 (and ungraded spots)
    correct: bool | None
    leak_key: str
    graded: bool               # False == no reference available (report: partial)
    frequency: float | None = None   # chosen freq (t1/2) or population freq (t3)
    flags: list[str] = field(default_factory=list)
    note: str = ""
    # What the grading solution ASSUMED, carried from Solution.range_ctx.
    # PLAN §5.3 promises the tier-2 range approximation is recorded "in each
    # Solution's provenance AND in the leak report"; the first half held and the
    # second was dropped at persist, so a tier-2 grade reached the user as
    # "exact — solver-graded" with no sign that both ranges were uniform and the
    # stacks symmetric ([R4-1]). Tier 1 leaves it None: a chart grade's
    # provenance is the chart, and inventing a string here would make an absent
    # disclosure look like a present one.
    provenance: str | None = None


def _table_size(d: Decision) -> int:
    """Players dealt in, from the formation token decisions.py always writes
    (`"{n}max:{pos}"`). Malformed means a construction bug — refuse loudly,
    like the seat guard: a silent default would mis-key the grading."""
    head = d.formation.split("max")[0]
    if not head.isdigit():
        raise ValueError(f"malformed formation {d.formation!r}")
    return int(head)


def _jamfold_position(d: Decision) -> str | None:
    """'SB' / 'BB' if this is a chart-modelled push/fold spot, else None."""
    # NaN-safe bounds check (round-2 finding [E35]): `nan > 20.0` is False, so a
    # non-finite eff_bb used to sail through this gate into the chart solver.
    # Stating the admissible window positively rejects NaN as well as inf.
    if not (0.0 < d.eff_bb <= JAMFOLD_MAX_BB):
        return None
    if d.street != "preflop" or d.num_in_pot != 2:
        return None
    if d.position == "SB" and not d.opp_allin:
        return "SB"   # folded to SB, unopened vs BB
    if d.position == "BB" and d.opp_allin:
        return "BB"   # BB facing an all-in jam
    return None


def _chart_action(d: Decision, pos: str) -> str | None:
    """Map the hero's engine action onto the chart's action space."""
    label = d.chosen[0]
    if label == "fold":
        return "fold"
    if pos == "SB" and d.is_allin and label in ("bet", "raise", "allin"):
        return "jam"
    if pos == "BB" and label in ("call", "allin"):
        return "call"
    return None  # off-chart (e.g. a min-raise at push/fold depth)


def grade_tier1(d: Decision) -> Grading:
    pos = _jamfold_position(d)
    if pos is None:
        # preflop but off-chart (e.g. deep 3-bet): canonical postflop-style key,
        # keyed by the hero's action, no depth bucket.
        return Grading(d.index, TIER_CHART, d.action_type, None, None, None,
                       postflop_category(d.formation, d.street, d.action_type),
                       graded=False,
                       note="preflop but not a chart-modelled jam/fold spot")
    # Snap ONCE, here, and use that depth for both the category and the answer
    # key (round-2 finding [E33]). Keying the leak off the snapped bucket while
    # solving the chart at the RAW eff_bb made the grader and the drill that
    # category selects disagree — at 12.5bb the two charts flip jam/fold on 21
    # hand classes, so a hand could be marked a leak and the drill that trains
    # it prescribe the opposite action. The bucket is the syllabus unit; the
    # drill generator already solves at it (drills/generator.py:jamfold_drills).
    depth_bb = float(snap_depth(d.eff_bb))
    # Same snap-ONCE discipline on the ante axis (round-3 finding [E49]): the
    # key carried no ante at all while the answer key below was solved at the
    # hand's RAW ante, so grader and drill disagreed for essentially every real
    # MTT hand. Both now key off this one snapped value.
    ante_bb = snap_ante(d.ante_bb)
    size = _table_size(d)
    if pos == "SB" and size > 2:
        # The seam ring.py's docstring recorded, closed on the grader side
        # too (rev 3.4): folded-to-SB at an n-handed table has n antes of
        # dead money in the pot — the HU chart priced 2. The ring artifact
        # now certifies every size 3–9, so the hand keys to ITS OWN table.
        from pokerlab.charts.ring import ring_range

        leak_key = ring_category("SB", depth_bb, ante_bb=ante_bb,
                                 table_size=size)
        ca = _chart_action(d, pos)
        if ca is None:
            return Grading(d.index, TIER_CHART, d.action_type, None, None,
                           None, leak_key, graded=False,
                           note="off-chart action")
        sol = ring_range("SB", depth_bb, ante_bb,
                         table_size=size)[hand_label(d.hole)]
        sc = score(sol, ca, d.pot_bb)
        return Grading(d.index, TIER_CHART, ca, sc.best_action,
                       sc.ev_loss_bb, sc.correct, leak_key, graded=True,
                       frequency=sc.chosen_frequency,
                       provenance=sol.range_ctx)
    leak_key = jamfold_category(pos, depth_bb, ante_bb=ante_bb)
    ca = _chart_action(d, pos)
    if ca is None:
        return Grading(d.index, TIER_CHART, d.action_type, None, None, None,
                       leak_key, graded=False, note="off-chart action")
    # The OTHER half of the seam, disclosed per-hand rather than closed: a
    # BB defend at a bigger table still keys to the HU chart, because the
    # ring defense key needs the JAMMER's position and Decision does not
    # carry it yet. A stated approximation, never a silent one.
    note = ("defend keyed to the HU chart — the ring defense key needs the "
            "jammer's position, which imported decisions do not carry yet"
            if pos == "BB" and size > 2 else "")
    sol = jamfold_range(pos, depth_bb, ante_bb)[hand_label(d.hole)]
    sc = score(sol, ca, d.pot_bb)
    return Grading(d.index, TIER_CHART, ca, sc.best_action, sc.ev_loss_bb,
                   sc.correct, leak_key, graded=True,
                   frequency=sc.chosen_frequency,
                   provenance=sol.range_ctx, note=note)


def grade_tier2(d: Decision, solution: Solution | None) -> Grading:
    leak_key = postflop_category(d.formation, d.street, d.action_type)
    if solution is None:
        return Grading(d.index, TIER_SOLVER, d.action_type, None, None, None,
                       leak_key, graded=False,
                       note="no solution in library — queued for solve")
    sc = score(solution, d.action_type, d.pot_bb)
    return Grading(d.index, TIER_SOLVER, d.action_type, sc.best_action,
                   sc.ev_loss_bb, sc.correct, leak_key, graded=True,
                   frequency=sc.chosen_frequency,
                   provenance=solution.range_ctx)


def grade_tier3(d: Decision, population: Population) -> Grading:
    """Multiway postflop: NEVER an ev_loss — only a frequency-deviation flag.

    Always a completed (approx) grading: the decision is routed and recorded
    with ev_loss=None. A population baseline is optional metadata — present it
    flags rare actions; absent it just leaves `frequency` None.
    """
    freq = population.frequency(d.formation, d.street, d.action_type)
    flags: list[str] = []
    note = ""
    if freq is None:
        note = "no population baseline for spot"
    elif freq < RARE_ACTION_FREQ:
        flags.append("freq_deviation")
        note = f"population plays {d.action_type} only {freq:.0%} here"
    return Grading(d.index, TIER_BEST_AVAILABLE, d.action_type, None, None, None,
                   postflop_category(d.formation, d.street, d.action_type),
                   graded=True, frequency=freq, flags=flags, note=note)


def grade_decision(d: Decision, *, population: Population,
                   solution: Solution | None = None) -> Grading:
    if d.tier == TIER_CHART:
        return grade_tier1(d)
    if d.tier == TIER_SOLVER:
        return grade_tier2(d, solution)
    return grade_tier3(d, population)
