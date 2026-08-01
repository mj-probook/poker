"""Multiway postflop drills — tier 3 (population-graded), advisory-guided.

THE HONESTY RULE THIS MODULE EXISTS AROUND (CLAUDE.md; plan §5.3): multiway
postflop has no trustworthy EV reference — the in-house solvers are HU
machines, and a multiway equilibrium is neither computed nor claimed. So
these drills carry NO action EVs at all (`Solution.actions == {}`): grading
is `hh.grade_tier3`'s contract — the chosen action TYPE compared to the
versioned population table (`population/default.toml`), rare actions
flagged, nothing else claimed. No correct/incorrect. No ev_loss. Ever.

What the built machinery CAN honestly contribute is ADVISORY analysis,
rendered as clearly-labeled study aids, never grades:

  * river spots ship a CERTIFIED HU-collapsed solve — both villains merged
    into ONE opponent, ranges uniform (the tier-2 approximation contract,
    plan §5.3 rev-3.2: the solve is exact, the inputs are the
    approximation), measured Nash gap ≤ 0.01bb, re-certified by the fast
    suite from the stored strategies like every other artifact;
  * flop and river spots ship seeded Monte Carlo equity vs the two-villain
    field (n and seed disclosed — a flop SOLVE would take ~30s/iteration
    and a half-converged key is a fabricated one, so no flop solve ships).

The scene (disclosed in every range_ctx): 6-max table, BTN opens 2.5bb, CO
and BB call, SB folds — pot 8.0bb, 37.5bb behind, three players to the
flop, checked through to the drilled street, BB (hero) first to act.
Population key "6max:BB" — the table's own vocabulary.

The M5/M7 ReBeL value nets contribute NOTHING here on purpose: the M7 spike
is a recorded NO-GO (docs/notes/tiny-hunl-spike.md) and numbers from a
machine that failed its own accuracy audit do not belong in a training UI,
advisory label or not.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from pokerlab.charts import hands
from pokerlab.drills.categories import postflop_category
from pokerlab.drills.generator import Drill
from pokerlab.drills.river import PRETTY, RIVER_CFG, river_boards
from pokerlab.engine.cards import card_from_str, card_to_str
from pokerlab.solver import subgame as sg
from pokerlab.types import TIER_BEST_AVAILABLE, Solution

DATA_PATH = Path(__file__).resolve().parents[1] / "solver" / "data" / \
    "multiway_advisory.npz"

MW_TABLE = ("BB", "CO", "BTN")     # postflop action order, 3 to the flop
MW_POT = 8.0                       # 3 × 2.5bb open + folded SB 0.5
MW_STACK = 37.5                    # 40bb - the 2.5bb call
MW_POPULATION_KEY = "6max:BB"      # the population table's own vocabulary
MW_TARGET_GAP = 0.01
EQUITY_SAMPLES = 2000

_NOTE = ("Multiway is tier 3: no EV grading exists for this spot by design "
         "— feedback is the population frequency of your action type, plus "
         "advisory analysis that never grades.")

_CAVEAT = ("Advisory only — HU-collapsed solve: both villains merged into "
           "ONE opponent, ranges uniform. The multiway equilibrium is NOT "
           "solved; these numbers guide study, they never grade.")


def _ctx(street: str) -> str:
    return (f"population:multiway:{MW_POPULATION_KEY}:{street}"
            f":line=BTN2.5-CO-call-BB-call-checked-through"
            f":graded=action-type-frequency-only")


def _cards(s: str) -> tuple[int, ...]:
    return tuple(card_from_str(s[i:i + 2]) for i in range(0, len(s), 2))


def hero_combo(cls: str, board: tuple[int, ...]) -> tuple[int, int] | None:
    """Deterministic concrete combo for a class: first one off the board."""
    for a, b in hands.card_combos(cls):
        if a not in board and b not in board:
            return (a, b)
    return None


def equity_seed(board_str: str, cls: str) -> int:
    import zlib
    return zlib.crc32(f"{board_str}:{cls}".encode())


def mc_equity_vs_field(hero: tuple[int, int], board: tuple[int, ...],
                       seed: int, n: int = EQUITY_SAMPLES) -> float:
    """Seeded Monte Carlo equity vs TWO uniform villains, board completed
    uniformly when on the flop. Disclosed as MC in the advisory note."""
    from pokerlab.engine.evaluator import rank_showdown

    rng = np.random.default_rng(seed)
    deck = np.array([c for c in range(52)
                     if c not in hero and c not in board], dtype=np.int64)
    need = 4 + (5 - len(board))
    total = 0.0
    for _ in range(n):
        draw = rng.choice(deck, size=need, replace=False)
        v1, v2 = (int(draw[0]), int(draw[1])), (int(draw[2]), int(draw[3]))
        full = board + tuple(int(c) for c in draw[4:])
        rh = rank_showdown(hero, full)
        r1, r2 = rank_showdown(v1, full), rank_showdown(v2, full)
        best = min(rh, r1, r2)                 # phevaluator: lower is better
        if rh == best:
            total += 1.0 / [rh, r1, r2].count(best)
    return total / n


def build_advisory_solver(board_str: str) -> sg.SubgameSolver:
    """HU-collapsed river solver: uniform vs uniform at the multiway pot."""
    board = _cards(board_str)
    ones = np.ones(sg.NUM_COMBOS)
    tree = sg.build_tree(board, pot0=MW_POT, stack=MW_STACK, cfg=RIVER_CFG)
    return sg.SubgameSolver(tree, board, ones.copy(), ones.copy(),
                            pot0=MW_POT)


@lru_cache(maxsize=1)
def load_multiway_advisory(path: str | None = None):
    """{board5: (solver, stored avg, stored gap)} plus equities via
    `load_equities` — the same rebuild-and-re-certify pattern as river.py."""
    p = Path(path) if path else DATA_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"multiway advisory artifact missing: {p}\n"
            "generate it with: uv run python scripts/gen_multiway_advisory.py")
    out = {}
    with np.load(p) as d:
        boards = sorted({name.split("/")[1] for name in d.files
                         if name.startswith("solve/")})
        for b in boards:
            solver = build_advisory_solver(b)
            avg = [d[f"solve/{b}/avg/{i}"].astype(np.float64)
                   for i in range(len(solver.decisions))]
            out[b] = (solver, avg, float(d[f"solve/{b}/meta"][0]))
    return out


@lru_cache(maxsize=1)
def load_equities(path: str | None = None) -> dict[str, np.ndarray]:
    """{f"{board_str}|{street}": (169,) MC equity vs the field}."""
    p = Path(path) if path else DATA_PATH
    with np.load(p) as d:
        return {name.split("/", 1)[1].replace("/", "|"): d[name]
                for name in d.files if name.startswith("eq/")}


def _drill(street: str, board: tuple[int, ...], board_str: str, tex: str,
           cls: str, hero: tuple[int, int], eq: float,
           advisory: dict) -> Drill:
    board_cards = tuple(card_to_str(c) for c in board)
    hero_cards = (card_to_str(hero[0]), card_to_str(hero[1]))
    leak = postflop_category(f"mw3.BTNCOBB.{tex}", street, "root")
    where = ("checked to you on the flop" if street == "flop"
             else "checked through to the river")
    return Drill(
        drill_id=f"{leak}:{cls}@{board_str}",
        kind="multiway", position="BB", depth_bb=MW_STACK, hand_label=cls,
        solution=Solution(actions={}, range_ctx=_ctx(street),
                          source="population"),
        pot_bb=MW_POT, leak_key=leak,
        legal_actions=("check", "bet"),
        description=(f"Multiway {street} {' '.join(board_cards)} — BTN "
                     f"opened 2.5bb, CO and BB called (6-max), {where}, "
                     f"{hero_cards[0]} {hero_cards[1]}: your action?"),
        action_note=_NOTE,
        tier=TIER_BEST_AVAILABLE,
        board=board_cards, hero_cards=hero_cards, table=MW_TABLE,
        population_key=MW_POPULATION_KEY,
        advisory=advisory,
    )


@lru_cache(maxsize=1)
def _multiway_drills_cached() -> tuple[Drill, ...]:
    return tuple(_build_multiway_drills())


def multiway_drills() -> list[Drill]:
    return list(_multiway_drills_cached())


def _build_multiway_drills() -> list[Drill]:
    eqs = load_equities()
    solves = load_multiway_advisory()
    eq_note = (f"Monte Carlo vs two uniform hands, n={EQUITY_SAMPLES}, "
               f"seed=crc32(board:class) — advisory, not a grade")
    out: list[Drill] = []
    textures = dict(river_boards())
    for board5, tex in river_boards():
        flop_str = board5[:6]
        flop = _cards(flop_str)
        river = _cards(board5)
        solver, avg, gap = solves[board5]
        labels, evs, freqs, _ = solver.root_action_evs(0, avg=avg)
        live_pos = {full: i for i, full in enumerate(solver.live.tolist())}
        for ci, cls in enumerate(hands.HAND_CLASSES):
            hf = hero_combo(cls, flop)
            if hf is not None:
                out.append(_drill(
                    "flop", flop, flop_str, tex, cls, hf,
                    float(eqs[f"{flop_str}|flop"][ci]),
                    {"caveat": _CAVEAT,
                     "equity_vs_field": float(eqs[f"{flop_str}|flop"][ci]),
                     "equity_note": eq_note}))
            hr = hero_combo(cls, river)
            if hr is None:
                continue
            key = (min(hr), max(hr))
            li = live_pos.get(sg.COMBO_INDEX[key])
            if li is None:
                continue
            solve = {PRETTY[lab]: round(float(evs[j][li]), 3)
                     for j, lab in enumerate(labels)}
            out.append(_drill(
                "river", river, board5, tex, cls, hr,
                float(eqs[f"{board5}|river"][ci]),
                {"caveat": _CAVEAT, "solve": solve, "gap": round(gap, 4),
                 "equity_vs_field": float(eqs[f"{board5}|river"][ci]),
                 "equity_note": eq_note}))
    return out
