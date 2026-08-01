"""Full-ring first-in jam/fold chain charts (plan §5.1 extension).

The game: a first-in jammer at a 9-max position either open-jams or folds;
every player still to act behind responds call-or-fold in order; the first
call ends the hand at a showdown. Chip-EV, symmetric stacks, per-player ante.

Three deliberate restrictions, all DISCLOSED in every Solution's range_ctx
because an answer key must state the game it actually solved:

  * single-caller — overcall trees are not modeled (the standard push/fold
    chart restriction; multiway all-ins are rare and pricing them would need
    a different tree);
  * pairwise card removal — exact jammer↔responder removal via the same
    `joint_prior`, but responder↔responder removal is ignored (fold-through
    probabilities condition on the jammer's cards only);
  * symmetric stacks — every seat plays `depth_bb`, like the HU charts.

Within that restricted game the solve carries the SAME certificate as the HU
charts: measured mean best-response gain (`exploitability`) — recomputable
from the strategies alone, which is how the cache test re-certifies every
shipped chart from scratch.

Anchor: a 2-seat table (SB jammer, BB responder) is exactly the game
`solve_jamfold` solves, and the chain solver reproduces it (test_charts_ring).

Modeling seam, stated rather than hidden: the HU `SBjam`/`BBcall` charts model
a 2-player table, so their ante variants price 2 antes of dead money. A real
9-max hand folded to the SB has 9 antes in the pot. The ring model prices
folded antes correctly for its own formations (UTG..BTN); reconciling the SB
formation with full-ring dead money is future work, not silently changed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import hands
from .equity import load_equity_matrix
from .jamfold import _validate_ante, _validate_depth, joint_prior

N = 169

# 9-max preflop action order. First-in means everyone before the jammer
# folded; blinds act last, so folded-before seats never include a blind and
# their dead money is antes only.
RING_ORDER: tuple[str, ...] = (
    "UTG", "UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB", "BB")
BLINDS: dict[str, float] = {"SB": 0.5, "BB": 1.0}

# Positions the drill grid solves as first-in jammers. SB is here TOO, under
# its own `.9max`-keyed formation: folded-to-SB at nine dealt players prices
# seven extra antes of dead money the HU `SBjam` chart never modeled — the
# seam the module docstring documents, closed on the drill side. (The HH
# grader still keys real SB hands to the HU chart; re-keying it needs a
# measured table-size bucket design, not a silent swap.)
RING_JAMMERS: tuple[str, ...] = ("UTG", "UTG1", "UTG2", "LJ", "HJ", "CO",
                                 "BTN", "SB")

DATA_PATH = Path(__file__).resolve().parent / "data" / "ring_charts.npz"

# Table sizes the drill grid ships (players dealt in). Every size is the
# button-anchored SUFFIX of the 9-max order: position names keep their 9-max
# labels (6-max first-in is "LJ") because distance-to-button is the strategic
# fact, and every range_ctx states the actual size (`ring6`). 2 is the HU
# table (the jamfold module's game); 9 is the classic full ring.
TABLE_SIZES: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9)


def table_for_size(n: int) -> tuple[str, ...]:
    """Button-anchored suffix of RING_ORDER for an n-handed table."""
    if n not in TABLE_SIZES:
        raise ValueError(f"table size {n} not in {TABLE_SIZES}")
    return RING_ORDER[len(RING_ORDER) - n:]


@dataclass(frozen=True)
class RingSolution:
    """Chain solve for one (table, jammer, depth, ante) formation."""

    table: tuple[str, ...]
    jammer: str
    depth_bb: float
    ante: float
    jam: np.ndarray                    # (169,) jammer jam frequency
    jam_ev: np.ndarray                 # (169,) per-hand EV of jamming (bb)
    jam_fold_ev: float
    calls: dict[str, np.ndarray]       # responder -> (169,) call frequency
    call_ev: dict[str, np.ndarray]     # responder -> per-hand EV of calling
    call_fold_ev: dict[str, float]
    exploitability: float              # mean best-response gain per player, bb

    def jam_combos(self) -> float:
        return float(sum(f * hands.combos(h)
                         for f, h in zip(self.jam, hands.HAND_CLASSES)))


def _validate_table(table: tuple[str, ...], jammer: str) -> None:
    if any(p not in RING_ORDER for p in table):
        unknown = [p for p in table if p not in RING_ORDER]
        raise ValueError(f"unknown position(s) {unknown!r}; valid: {RING_ORDER}")
    order = [RING_ORDER.index(p) for p in table]
    if len(set(table)) != len(table) or order != sorted(order):
        raise ValueError(f"table must be unique positions in action order, "
                         f"got {table!r}")
    if "SB" not in table or "BB" not in table:
        raise ValueError(f"a table needs both blinds, got {table!r}")
    if jammer not in table:
        raise ValueError(f"jammer {jammer!r} not at table {table!r}")
    if jammer == "BB":
        raise ValueError("BB cannot open-jam first-in — if everyone folds to "
                         "the BB there is no decision to train")


def _regret_match(r0: np.ndarray, r1: np.ndarray) -> np.ndarray:
    p0 = np.maximum(r0, 0.0)
    p1 = np.maximum(r1, 0.0)
    tot = p0 + p1
    return np.where(tot > 0.0, p0 / np.where(tot > 0.0, tot, 1.0), 0.5)


class _Chain:
    """Precomputed constants + counterfactual values for one formation."""

    def __init__(self, table: tuple[str, ...], jammer: str,
                 s: float, a: float, E: np.ndarray, P: np.ndarray):
        p = table.index(jammer)
        self.responders = list(table[p + 1:])
        n = len(table)
        b_total = sum(BLINDS.get(q, 0.0) for q in table)
        b_jam = BLINDS.get(jammer, 0.0)
        self.P = P
        self.w = P.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.Pcond = np.where(self.w[:, None] > 0, P / self.w[:, None], 0.0)
        self.fold_ev = -(b_jam + a)
        self.win_all = (b_total - b_jam) + (n - 1) * a
        # Per-responder showdown payoff matrices. Dead money D_q: the blinds
        # not held by the two showdown players plus the other seats' antes.
        self.jam_pay: dict[str, np.ndarray] = {}
        self.call_pay: dict[str, np.ndarray] = {}
        self.resp_fold: dict[str, float] = {}
        for q in self.responders:
            d_q = (b_total - b_jam - BLINDS.get(q, 0.0)) + (n - 2) * a
            pot = 2.0 * s + d_q
            self.jam_pay[q] = E * pot - s            # jammer's net, [i, c]
            self.call_pay[q] = E.T * pot - s         # caller's net,  [i, c]
            self.resp_fold[q] = -(BLINDS.get(q, 0.0) + a)

    def values(self, x: np.ndarray, ys: dict[str, np.ndarray]):
        """Counterfactual (chance-weighted) action values for every player."""
        # fold-through of responder k given the jammer's class (pairwise
        # removal: conditions on the jammer's cards only — see module doc)
        F = {k: self.Pcond @ (1.0 - ys[k]) for k in self.responders}
        v_call: dict[str, np.ndarray] = {}
        v_rfold: dict[str, np.ndarray] = {}
        reach = np.ones(N)                           # prefix Π F_k over i
        v_jam_cond = np.zeros(N)
        for q in self.responders:
            # jammer's showdown-vs-q term, conditioned on their own class
            t_q = ((self.Pcond * self.jam_pay[q]) @ ys[q])
            v_jam_cond += reach * t_q
            # responder q's node: jammer jammed, everyone before q folded
            a_q = self.P * (x * reach)[:, None]
            v_call[q] = (a_q * self.call_pay[q]).sum(axis=0)
            v_rfold[q] = a_q.sum(axis=0) * self.resp_fold[q]
            reach = reach * F[q]
        v_jam_cond += reach * self.win_all           # everyone folded
        v_jam = self.w * v_jam_cond
        v_jfold = self.w * self.fold_ev
        return v_jam, v_jfold, v_call, v_rfold


def _nash_gap(chain: _Chain, x: np.ndarray, ys: dict[str, np.ndarray]) -> float:
    """Mean per-player best-response gain — the certificate, recomputable
    from the strategies alone (no solver state)."""
    v_jam, v_jfold, v_call, v_rfold = chain.values(x, ys)
    on = x * v_jam + (1.0 - x) * v_jfold
    gain = float((np.maximum(v_jam, v_jfold) - on).sum())
    for q in chain.responders:
        on_q = ys[q] * v_call[q] + (1.0 - ys[q]) * v_rfold[q]
        gain += float((np.maximum(v_call[q], v_rfold[q]) - on_q).sum())
    return gain / (1 + len(chain.responders))


@lru_cache(maxsize=None)
def solve_ring(jammer: str, depth_bb: float, ante: float = 0.0,
               table: tuple[str, ...] = RING_ORDER,
               iters: int = 2000, target_gap: float = 1e-5,
               max_iters: int = 60000) -> RingSolution:
    """Solve one first-in chain formation (cached). CFR+ with simultaneous
    updates and linear averaging, like the HU solver it reduces to.

    Runs in chunks of `iters` until the measured Nash gap of the AVERAGED
    profile is ≤ `target_gap` (or `max_iters` is hit). A fixed budget left two
    shallow-ante formations (UTG/UTG1 at 5bb, 0.125bb ante — maximal
    near-indifference) at ~2e-4 while the rest of the grid sat at ~e-6; the
    certificate is the stopping rule, not a hope, so the solver iterates until
    it holds. Deterministic: same inputs, same chunk boundaries, same output.
    """
    _validate_depth(depth_bb)
    _validate_ante(ante)
    _validate_table(table, jammer)
    E = load_equity_matrix().equity_matrix
    P = joint_prior()
    chain = _Chain(table, jammer, float(depth_bb), float(ante), E, P)
    resp = chain.responders

    r_jam = np.zeros((2, N))
    r_resp = {q: np.zeros((2, N)) for q in resp}
    s_jam = np.zeros((2, N))
    s_resp = {q: np.zeros((2, N)) for q in resp}
    t = 0
    while True:
        for _ in range(iters):
            t += 1
            x = _regret_match(r_jam[0], r_jam[1])
            ys = {q: _regret_match(r_resp[q][0], r_resp[q][1]) for q in resp}
            v_jam, v_jfold, v_call, v_rfold = chain.values(x, ys)

            v = x * v_jam + (1.0 - x) * v_jfold
            r_jam[0] += v_jam - v
            r_jam[1] += v_jfold - v
            np.maximum(r_jam, 0.0, out=r_jam)
            s_jam[0] += t * x
            s_jam[1] += t * (1.0 - x)
            for q in resp:
                vq = ys[q] * v_call[q] + (1.0 - ys[q]) * v_rfold[q]
                r_resp[q][0] += v_call[q] - vq
                r_resp[q][1] += v_rfold[q] - vq
                np.maximum(r_resp[q], 0.0, out=r_resp[q])
                s_resp[q][0] += t * ys[q]
                s_resp[q][1] += t * (1.0 - ys[q])

        x = s_jam[0] / (s_jam[0] + s_jam[1])
        ys = {q: s_resp[q][0] / (s_resp[q][0] + s_resp[q][1]) for q in resp}
        if _nash_gap(chain, x, ys) <= target_gap or t >= max_iters:
            break

    v_jam, _, v_call, _ = chain.values(x, ys)
    with np.errstate(invalid="ignore", divide="ignore"):
        jam_ev = np.where(chain.w > 0, v_jam / chain.w, 0.0)
        call_ev = {}
        reach = np.ones(N)
        for q in resp:
            F_q = chain.Pcond @ (1.0 - ys[q])
            node = (chain.P * (x * reach)[:, None]).sum(axis=0)
            call_ev[q] = np.where(node > 0, v_call[q] / node, 0.0)
            reach = reach * F_q

    return RingSolution(
        table=table, jammer=jammer, depth_bb=float(depth_bb),
        ante=float(ante), jam=x, jam_ev=jam_ev, jam_fold_ev=chain.fold_ev,
        calls=ys, call_ev=call_ev, call_fold_ev=dict(chain.resp_fold),
        exploitability=_nash_gap(chain, x, ys),
    )


# --------------------------------------------------------------------------- #
# Checked-in artifact (the equity-matrix pattern). Solving the 105-formation
# drill grid takes ~76s, so the charts ship pre-solved; the cache test
# re-certifies every stored formation's Nash gap from scratch on every run,
# so the artifact can never silently rot into a black box.
# --------------------------------------------------------------------------- #
def cache_key(jammer: str, depth_bb: float, ante: float,
              table_size: int = 9) -> str:
    """9-max keys stay byte-identical (orphan rule); short tables carry a
    `.{n}max` suffix on the jammer token, mirroring the category tokens."""
    tag = "" if table_size == 9 else f".{table_size}max"
    return f"{jammer}{tag}|{depth_bb:g}|{ante:g}"


def _parse_jammer_token(token: str) -> tuple[str, int]:
    """"CO" -> ("CO", 9); "CO.6max" -> ("CO", 6)."""
    if "." in token:
        jammer, tag = token.split(".", 1)
        return jammer, int(tag.removesuffix("max"))
    return token, 9


@lru_cache(maxsize=1)
def load_ring_charts(path: str | None = None) -> dict[str, RingSolution]:
    p = Path(path) if path else DATA_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"ring chart artifact missing: {p}\n"
            "generate it with: uv run python scripts/gen_ring_charts.py")
    out: dict[str, RingSolution] = {}
    with np.load(p) as d:
        keys = sorted({name.split("/")[0] for name in d.files})
        for key in keys:
            jammer, size = _parse_jammer_token(key.split("|")[0])
            table = table_for_size(size)
            behind = table[table.index(jammer) + 1:]
            depth, ante, jam_fold_ev, expl = (float(v) for v in d[f"{key}/meta"])
            calls_a = d[f"{key}/calls"]
            call_ev_a = d[f"{key}/call_ev"]
            call_fold_a = d[f"{key}/call_fold"]
            out[key] = RingSolution(
                table=table, jammer=jammer, depth_bb=depth, ante=ante,
                jam=d[f"{key}/jam"], jam_ev=d[f"{key}/jam_ev"],
                jam_fold_ev=jam_fold_ev,
                calls={q: calls_a[i] for i, q in enumerate(behind)},
                call_ev={q: call_ev_a[i] for i, q in enumerate(behind)},
                call_fold_ev={q: float(call_fold_a[i])
                              for i, q in enumerate(behind)},
                exploitability=expl,
            )
    return out


def ring_solution(jammer: str, depth_bb: float, ante: float = 0.0,
                  table_size: int = 9) -> RingSolution:
    """A formation's solve: from the checked-in artifact when present (the
    entire drill grid is), a live `solve_ring` otherwise — same solver, same
    iteration budget, so the two paths agree (pinned by the slow cache test)."""
    cached = load_ring_charts().get(
        cache_key(jammer, depth_bb, ante, table_size))
    return cached if cached is not None else solve_ring(
        jammer, depth_bb, ante, table=table_for_size(table_size))


# --------------------------------------------------------------------------- #
# Normalized-Solution API (what the drill generator consumes). The three
# model restrictions are DISCLOSED in every range_ctx — the answer key states
# the game it solved, it does not let a doc do it.
# --------------------------------------------------------------------------- #
def _ctx(sol: RingSolution) -> str:
    return (f"chart:ring{len(sol.table)}:first-in:{sol.jammer}:"
            f"{sol.depth_bb:g}bb:a{sol.ante:g}:"
            f"single-caller|pairwise-removal|stacks-symmetric")


def ring_range(jammer: str, depth_bb: float, ante: float = 0.0,
               table_size: int = 9) -> dict[str, "Solution"]:
    """{hand_label: Solution} for the first-in jammer's jam/fold decision."""
    from pokerlab.types import Solution

    sol = ring_solution(jammer, depth_bb, ante, table_size)
    ctx = _ctx(sol)
    return {
        label: Solution(
            actions={"jam": (float(sol.jam_ev[i]), float(sol.jam[i])),
                     "fold": (float(sol.jam_fold_ev),
                              1.0 - float(sol.jam[i]))},
            range_ctx=ctx, source="chart")
        for i, label in enumerate(hands.HAND_CLASSES)
    }


def ring_defense_range(position: str, *, versus: str, depth_bb: float,
                       ante: float = 0.0,
                       table_size: int = 9) -> dict[str, "Solution"]:
    """{hand_label: Solution} for `position` defending call/fold against a
    first-in jam from `versus`."""
    from pokerlab.types import Solution

    sol = ring_solution(versus, depth_bb, ante, table_size)
    if position not in sol.calls:
        raise ValueError(
            f"{position!r} is not behind a first-in {versus!r} jam — "
            f"responders are {sorted(sol.calls)}")
    y, ev = sol.calls[position], sol.call_ev[position]
    fold_ev = sol.call_fold_ev[position]
    ctx = _ctx(sol)
    return {
        label: Solution(
            actions={"call": (float(ev[i]), float(y[i])),
                     "fold": (float(fold_ev), 1.0 - float(y[i]))},
            range_ctx=ctx, source="chart")
        for i, label in enumerate(hands.HAND_CLASSES)
    }
