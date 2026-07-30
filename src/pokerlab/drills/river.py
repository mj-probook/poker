"""River drills — the first honest postflop slice (tier 2, solver-graded).

Why the river and not the flop: a river subgame has no chance nodes, so the
in-house subgame solver (solver/subgame.py, the M3 tier-2 engine) solves it
to a MEASURED Nash gap in seconds — flop/turn trees cost ~30s per CFR
iteration and belong to the nightly batch drain, not to a checked-in answer
key. Shipping a half-converged flop artifact would be a fabricated key
wearing a real one's clothes; the river ships with a certificate.

The game, all inputs disclosed in every Solution's range_ctx (plan §5.3:
the solve is exact — the INPUTS are the approximation):

  * BTN-open / BB-call single-raised pot from the flops25 fixture
    (self-authored ranges, pot 5bb, 37.5bb behind);
  * the CHECKED-DOWN line: flop and turn went check/check, so both ranges
    reach the river UNFILTERED — a stated modeling choice, not a claim that
    real play checks down;
  * bet grid 33% / 75% / all-in, raises capped at 2 — sizes outside the
    grid are not priced (the drill page says so in the action note).

Boards: for each of the 8 flop textures, up to two flops25 entries extended
by a DETERMINISTIC runout (fixed offsets into the remaining deck, stated in
`_runout`). The full board is always shown; nothing about the scenario is
hidden. Drills present a CONCRETE combo (the class' highest-weight legal
one) and grade it against that combo's OWN solver EVs — no class averaging,
because on a real board the suits are the strategy.

The artifact stores the full average-strategy set per board, so the fast
suite re-certifies every shipped board's Nash gap from scratch via
`SubgameSolver.exploitability(avg=stored)` — the same
no-black-box rule as the chart artifacts.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from pokerlab.charts import hands
from pokerlab.drills.categories import postflop_category
from pokerlab.drills.generator import Drill
from pokerlab.engine.cards import card_from_str, card_to_str
from pokerlab.solver import flops25, subgame as sg
from pokerlab.solver.subgame import COMBO_INDEX
from pokerlab.types import TIER_SOLVER, Solution

DATA_PATH = Path(__file__).resolve().parents[1] / "solver" / "data" / \
    "river_strategies.npz"

RIVER_CFG = sg.BetConfig(sizes=(0.33, 0.75), jam=True, max_raises=2)
POT_BB = 5.0          # flops25 SRP pot
EFF_BB = 37.5         # behind after the open/call
TARGET_GAP_BB = 0.01  # solver stopping rule; ε floor is 0.1bb — 10x inside
BOARDS_PER_TEXTURE = 2

# solver action label -> button label
PRETTY = {"check": "check", "bet_0.33": "bet 33%", "bet_0.75": "bet 75%",
          "jam": "jam"}

_NOTE = ("Bet sizes are the solved grid (33%, 75%, all-in); other sizes are "
         "not priced.")

# full combo index -> its two cards, inverse of COMBO_INDEX
_COMBO_CARDS = {v: k for k, v in COMBO_INDEX.items()}


def _cards(s: str) -> tuple[int, ...]:
    return tuple(card_from_str(s[i:i + 2]) for i in range(0, len(s), 2))


def _runout(flop: tuple[int, ...]) -> tuple[int, int]:
    """Deterministic turn/river: fixed offsets 5 and 29 into the sorted
    remaining deck. A scenario choice, not a claim — the board is shown."""
    remaining = sorted(set(range(52)) - set(flop))
    return remaining[5], remaining[29]


def river_boards() -> list[tuple[str, str]]:
    """[(board5_str, flop_texture)] — up to two flops per texture."""
    per: dict[str, int] = {}
    out = []
    for entry in flops25.load_flops25()["flops"]:
        tex = entry["texture"]
        if per.get(tex, 0) >= BOARDS_PER_TEXTURE:
            continue
        per[tex] = per.get(tex, 0) + 1
        flop = _cards(entry["cards"])
        t, r = _runout(flop)
        board = flop + (t, r)
        out.append(("".join(card_to_str(c) for c in board), tex))
    return out


def build_river_solver(board_str: str) -> sg.SubgameSolver:
    board = _cards(board_str)
    oop, ip = flops25.range_vectors()
    tree = sg.build_tree(board, pot0=POT_BB, stack=EFF_BB, cfg=RIVER_CFG)
    return sg.SubgameSolver(tree, board, oop, ip, pot0=POT_BB)


@lru_cache(maxsize=1)
def load_river_solves(path: str | None = None
                      ) -> dict[str, tuple[sg.SubgameSolver, list, float]]:
    """{board: (solver, stored avg strategies, stored gap)} — the solver is
    rebuilt deterministically; strategies come from the artifact."""
    p = Path(path) if path else DATA_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"river artifact missing: {p}\n"
            "generate it with: uv run python scripts/gen_river_drills.py")
    out = {}
    with np.load(p) as d:
        boards = sorted({name.split("/")[0] for name in d.files})
        for b in boards:
            gap = float(d[f"{b}/meta"][0])
            solver = build_river_solver(b)
            avg = [d[f"{b}/avg/{i}"].astype(np.float64)
                   for i in range(len(solver.decisions))]
            out[b] = (solver, avg, gap)
    return out


def _ctx(board: str, gap: float) -> str:
    return (f"solver:river:{board}:ranges=flops25-fixture-preflop-unfiltered"
            f":line=flop+turn-checked:grid=33/75/jam"
            f":measured_gap={gap:.4f}bb")


def river_drills() -> list[Drill]:
    """OOP (BB) root river decisions, one drill per hand class per board."""
    textures = dict(river_boards())
    out: list[Drill] = []
    for board, (solver, avg, gap) in load_river_solves().items():
        tex = textures[board]
        labels, evs, freqs, my_reach = solver.root_action_evs(0, avg=avg)
        valid_opp = solver.valid_opponent_reach(0)
        live_pos = {full: i for i, full in enumerate(solver.live.tolist())}
        ctx = _ctx(board, gap)
        leak = postflop_category(f"BTNvBB.{tex}", "river", "root")
        board_cards = tuple(board[i:i + 2] for i in range(0, len(board), 2))
        pretty_board = " ".join(board_cards)
        for cls in hands.HAND_CLASSES:
            best_i, best_w = None, 0.0
            for a, b in hands.card_combos(cls):
                i = live_pos.get(COMBO_INDEX[(min(a, b), max(a, b))])
                if i is not None and my_reach[i] > 0 and valid_opp[i] > 0:
                    if my_reach[i] > best_w:
                        best_i, best_w = i, float(my_reach[i])
            if best_i is None:
                continue        # class blocked or outside the BB range
            combo = _COMBO_CARDS[int(solver.live[best_i])]
            hero = (card_to_str(combo[0]), card_to_str(combo[1]))
            actions = {PRETTY[lab]: (float(evs[j][best_i]),
                                     float(freqs[j][best_i]))
                       for j, lab in enumerate(labels)}
            # id honors the "<leak_key>:<hand>" contract (a single colon —
            # consumers rsplit on it); the hand token carries the board too,
            # since one category serves two boards of the same texture
            out.append(Drill(
                drill_id=f"{leak}:{cls}@{board}",
                kind="river", position="BB", depth_bb=EFF_BB,
                hand_label=cls,
                solution=Solution(actions=actions, range_ctx=ctx,
                                  source="subgame_solver"),
                pot_bb=POT_BB, leak_key=leak,
                legal_actions=tuple(PRETTY[lab] for lab in labels),
                description=(f"River {pretty_board} — BTN vs BB "
                             f"single-raised pot, checked to the river, "
                             f"{hero[0]} {hero[1]}: your action?"),
                action_note=_NOTE,
                tier=TIER_SOLVER,
                board=board_cards, hero_cards=hero,
                table=("BB", "BTN"),
            ))
    return out
