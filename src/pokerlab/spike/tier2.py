"""M4 tier-2 activation — wire the Slice-D solver into the HH pipeline (Slice I).

Uses the seams Slice F already exposed, no changes to `hh` grading:

  * ``make_inline_solution_for(conn)`` -> a ``solution_for(decision)`` for
    `hh.report.grade_session`: resolve the decision's SpotKey, look it up in
    ``solution_index`` (a hit), load the cached solve via ``solve_io``, and
    return the hero's-hand-class `Solution` (solver bet-size labels collapsed to
    the HH vocabulary so `drills.scoring.score` can grade the hero's action).
  * ``make_drain_solver(conn)`` -> a ``solver(spot_key, decision)`` for
    `hh.persist.drain_batch_queue`: solve the decision's turn/river subgame with
    the real Slice-D CFR+ solver, cache + index it, and return the same
    collapsed hero `Solution`.

Scope guard (plan/impl §3 Slice I): only turn (4-card) and river (5-card)
subgames are solved — never a full flop — and only when the hero is the OOP root
actor of the subgame (the solved root's player). Other spots return None (miss).
"""

from __future__ import annotations

from collections.abc import Callable

from pokerlab.engine.cards import card_to_str
from pokerlab.hh.decisions import Decision, hand_label
from pokerlab.solver import subgame as sg
from pokerlab.solver.adapter import root_solution_by_class
from pokerlab.solver.solve_io import insert_solution_index, load_solve, write_solve
from pokerlab.solver.spotkey import resolve_spot_key
from pokerlab.store import db
from pokerlab.types import Solution, TIER_SOLVER

# tiny-HUNL betting shape: kept minimal here so a tier-2 grade solve is fast.
TIER2_CFG = sg.BetConfig(sizes=(0.75,), jam=True, max_raises=1)
DEFAULT_ITERS = 120

_DIRECT = {"check", "fold", "call", "jam"}


def spotkey_str(sk) -> str:
    """Canonical string for a flop-textured SpotKey (board_bucket|stack|formation)."""
    return f"{sk.board_bucket}|{sk.stack_bucket}|{sk.formation}"


def tier2_key(d: Decision) -> str:
    """Cache/index key for a turn/river tier-2 solve.

    SpotKey alone is street-agnostic (its board_bucket is the *flop* texture), so
    every decision on the same flop line would collide. Turn/river solves are
    board-specific, so we key by formation + stack bucket + the FULL visible
    board — distinct per street, no cross-street collision.
    """
    sk = resolve_spot_key(d.game_state)
    board = "".join(card_to_str(c) for c in d.board)
    return f"{sk.formation}|{sk.stack_bucket}bb|{board}"


def _hh_label(solver_label: str) -> str:
    if solver_label in _DIRECT:
        return solver_label
    if solver_label.startswith("raise"):
        return "raise"
    if solver_label.startswith("bet"):
        return "bet"
    return solver_label


def _collapse(actions: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    """Fold solver size-tagged labels into the HH action vocabulary.

    Frequencies sum; EV is the frequency-weighted mean (best-EV when a bucket is
    played with zero frequency), so the collapsed action is graded as "the pool
    of bets of that kind" — size-agnostic tier-2, honest for HH grading.
    """
    buckets: dict[str, list[tuple[float, float]]] = {}
    for label, (ev, freq) in actions.items():
        buckets.setdefault(_hh_label(label), []).append((float(ev), float(freq)))
    out: dict[str, tuple[float, float]] = {}
    for hh, lst in buckets.items():
        fsum = sum(f for _, f in lst)
        ev = (sum(ev * f for ev, f in lst) / fsum) if fsum > 0 else max(ev for ev, _ in lst)
        out[hh] = (ev, fsum)
    return out


def _collapsed_solution(sol: Solution) -> Solution:
    return Solution(actions=_collapse(sol.actions), range_ctx=sol.range_ctx,
                    source=sol.source)


def _solution_from_file(solve: dict, hand_cls: str) -> Solution | None:
    entry = solve.get("solutions", {}).get(hand_cls)
    if entry is None:
        return None
    actions = {k: tuple(v) for k, v in entry["actions"].items()}
    return _collapsed_solution(Solution(
        actions=actions, range_ctx=entry.get("range_ctx", ""),
        source=entry.get("source", "subgame_solver")))


def solvable(d: Decision) -> bool:
    """True iff this is a turn/river spot with the hero as OOP root actor."""
    return (d.tier == TIER_SOLVER and d.game_state is not None
            and len(d.board) >= 4 and d.position != "BTN")


def solve_decision(d: Decision, *, iters: int = DEFAULT_ITERS,
                   cfg: sg.BetConfig = TIER2_CFG) -> sg.SubgameSolver | None:
    """Solve the decision's turn/river subgame (uniform ranges). None if unsolvable."""
    if len(d.board) < 4:
        return None
    pot0 = max(d.pot_bb, 1.0)
    stack = max(d.eff_bb - pot0 / 2.0, pot0)  # approx effective remaining behind
    tree = sg.build_tree(tuple(d.board), pot0=pot0, stack=stack, cfg=cfg)
    solver = sg.SubgameSolver(tree, tuple(d.board), sg.uniform_range(),
                              sg.uniform_range(), pot0=pot0)
    if solver.decisions[0].player != 0:  # hero must be the OOP root actor
        return None
    solver.iterate(iters)
    return solver


def _hero_solution(d: Decision, solver: sg.SubgameSolver) -> Solution | None:
    ctx = f"solve|expl_bb={solver.exploitability():.4g}"
    per_class = root_solution_by_class(solver, player=0, range_ctx=ctx)
    sol = per_class.get(hand_label(d.hole))
    return _collapsed_solution(sol) if sol is not None else None


def make_inline_solution_for(conn) -> Callable[[Decision], Solution | None]:
    """solution_for for grade_session: a solution_index HIT -> hero Solution."""
    def solution_for(d: Decision) -> Solution | None:
        if d.tier != TIER_SOLVER or d.game_state is None:
            return None
        key = tier2_key(d)
        path = db.solution_path(conn, key)
        if path is None:
            return None
        return _solution_from_file(load_solve(path), hand_label(d.hole))
    return solution_for


def make_drain_solver(conn, *, iters: int = DEFAULT_ITERS, cfg: sg.BetConfig = TIER2_CFG,
                      persist: bool = True) -> Callable[[str, Decision], Solution | None]:
    """solver for drain_batch_queue: real solve, cache+index, hero Solution."""
    def solver(spot_key: str, d: Decision) -> Solution | None:
        if not solvable(d):
            return None
        try:
            solved = solve_decision(d, iters=iters, cfg=cfg)
            if solved is None:
                return None
            if persist:
                key = tier2_key(d)
                path, expl = write_solve(solved, key)
                insert_solution_index(conn, key, path, sg.SOLVER_VERSION,
                                      expl / solved.pot0)
            return _hero_solution(d, solved)
        except sg.DegenerateRangeError as exc:
            # No legal hero/villain matchup: re-solving cannot ever help, so
            # this is terminal for the row rather than a miss to retry — but it
            # must be visible as "unsolvable", not swallowed (wave-2 [E15][E25]).
            raise UnsolvableSpot(f"{tier2_key(d)}: {exc}") from exc
    return solver


def cache_solve(conn, d: Decision, *, iters: int = DEFAULT_ITERS,
                cfg: sg.BetConfig = TIER2_CFG) -> str | None:
    """Solve + write + index the decision's spot; return its SpotKey string."""
    solved = solve_decision(d, iters=iters, cfg=cfg)
    if solved is None:
        return None
    key = tier2_key(d)
    path, expl = write_solve(solved, key)
    insert_solution_index(conn, key, path, sg.SOLVER_VERSION, expl / solved.pot0)
    return key
