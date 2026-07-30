"""First-in OPEN game charts: fold / raise-to-r / jam, every action priced.

The jam/fold charts (jamfold.py, ring.py) solve a game whose only aggressive
action is the jam, which is why the drill page grades a raise as off-tree:
that game has no number for it. This module solves the bigger first-in game
where raises are real actions:

    opener:      fold | raise to r (r ∈ RAISE_SIZES) | jam
    each seat behind, in order:
        vs a jam:    call | fold          (first caller ends the hand)
        vs a raise:  re-jam | fold        (first re-jammer ends the chain)
    opener vs a re-jam:  call | fold

Every line terminates PREFLOP (fold chains, steals, all-in showdowns), so the
game is exactly solvable by the same vectorised CFR+ the other charts use and
carries the same certificate: the measured mean best-response gain, with the
solver iterating until it holds.

Disclosed restrictions, carried in every Solution's range_ctx because an
answer key states the game it solved:

  * single aggressive responder — no overcall / re-jam-over-re-jam trees;
  * no flat calls behind — calling a raise creates a postflop subgame this
    engine does not price (the same reason limp stays an off-tree distractor
    on the drill page);
  * pairwise card removal (as ring.py) and symmetric stacks.

Reduction anchor: with sizes=() the game IS the jam/fold chain, and the solve
must agree with `solve_ring` and certify under ring's own instrument
(test_charts_openraise).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import hands
from .equity import load_equity_matrix
from .jamfold import _validate_ante, _validate_depth, joint_prior
from .ring import BLINDS, RING_ORDER, _regret_match, _validate_table

N = 169

# Raise-to sizes in bb — exactly the buttons the drill page shows. Both are
# open sizes, not fractions, so they must stay below the stack they open.
RAISE_SIZES: tuple[float, ...] = (2.2, 3.0)


def sizes_for_depth(depth_bb: float) -> tuple[float, ...]:
    """The raise menu a depth actually supports. At 5bb it is EMPTY: raising
    to 2.2 commits 44% of the stack — push/fold theory's jam-only territory —
    and the solver agrees the actions are near-duplicates there in the most
    concrete way: the 5bb multi-action solves measurably fail to converge
    (Nash gap stuck at ~4e-3 after 90k iterations, vs ~1e-5 everywhere else)
    because CFR+ is being asked to separate options the game barely
    distinguishes. An open drill at 5bb is therefore an honest jam/fold
    decision, and no resteal spot exists there (no raise ever occurs)."""
    return () if depth_bb <= 5.0 else RAISE_SIZES


def _rlabel(r: float) -> str:
    return f"raise {r:g}bb"


@dataclass(frozen=True)
class OpenSolution:
    """One formation's solve of the open game."""

    table: tuple[str, ...]
    opener: str
    depth_bb: float
    ante: float
    sizes: tuple[float, ...]
    open_freq: dict[str, np.ndarray]     # action label -> (169,) frequency
    open_ev: dict[str, np.ndarray]       # action label -> (169,) per-hand EV
    # responder strategies, keyed (position, threat label). Aggressive action
    # is "call" vs a jam and "re-jam" vs a raise; EV is the aggressive line's.
    defend_freq: dict[tuple[str, str], np.ndarray]
    defend_ev: dict[tuple[str, str], np.ndarray]
    defend_fold_ev: dict[tuple[str, str], float]
    # opener's call-vs-re-jam, keyed (raise label, re-jammer position)
    callback_freq: dict[tuple[str, str], np.ndarray]
    exploitability: float                # mean BR gain per player, bb


def _regret_match_k(R: np.ndarray) -> np.ndarray:
    """(k, N) positive-regret matching; uniform where no positive regret."""
    pos = np.maximum(R, 0.0)
    tot = pos.sum(axis=0)
    k = R.shape[0]
    return np.where(tot > 0.0, pos / np.where(tot > 0.0, tot, 1.0), 1.0 / k)


class _OpenGame:
    """Precomputed constants + counterfactual values for one formation."""

    def __init__(self, table: tuple[str, ...], opener: str, s: float,
                 a: float, sizes: tuple[float, ...],
                 E: np.ndarray, P: np.ndarray):
        p = table.index(opener)
        self.responders = list(table[p + 1:])
        self.sizes = sizes
        self.threats = ["jam"] + [_rlabel(r) for r in sizes]
        self.r_of = {_rlabel(r): float(r) for r in sizes}
        n = len(table)
        b_total = sum(BLINDS.get(q, 0.0) for q in table)
        b_open = BLINDS.get(opener, 0.0)
        self.P = P
        self.w = P.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.Pcond = np.where(self.w[:, None] > 0, P / self.w[:, None], 0.0)
        self.fold_ev = -(b_open + a)
        self.ante = a
        self.steal = (b_total - b_open) + (n - 1) * a
        # per-responder showdown payoffs and dead money (same for a called
        # jam and a called re-jam: both are stacks-in with the others' blinds
        # and antes dead)
        self.jam_pay: dict[str, np.ndarray] = {}    # opener's net, [i, c]
        self.call_pay: dict[str, np.ndarray] = {}   # responder's net, [i, c]
        self.resp_fold: dict[str, float] = {}
        self.resteal_win: dict[tuple[str, str], float] = {}
        for q in self.responders:
            d_q = (b_total - b_open - BLINDS.get(q, 0.0)) + (n - 2) * a
            pot = 2.0 * s + d_q
            self.jam_pay[q] = E * pot - s
            self.call_pay[q] = E.T * pot - s
            self.resp_fold[q] = -(BLINDS.get(q, 0.0) + a)
            for r in sizes:
                # what the re-jammer wins when the opener folds: the raise,
                # the other blinds, the other antes
                self.resteal_win[(_rlabel(r), q)] = (
                    float(r) + (b_total - b_open - BLINDS.get(q, 0.0))
                    + (n - 1) * a)

    def values(self, x: dict[str, np.ndarray], ys: dict, zs: dict):
        """Counterfactual (chance-weighted) action values for every infoset.

        x: opener action label -> prob vec (labels: threats + "fold")
        ys: (q, threat) -> aggressive prob vec
        zs: (raise label, q) -> opener call-vs-re-jam prob vec
        """
        resp = self.responders
        v_open: dict[str, np.ndarray] = {}
        v_agg: dict[tuple[str, str], np.ndarray] = {}
        v_rfold: dict[tuple[str, str], np.ndarray] = {}
        v_zcall: dict[tuple[str, str], np.ndarray] = {}
        v_zfold: dict[tuple[str, str], np.ndarray] = {}
        for t in self.threats:
            is_jam = t == "jam"
            r = None if is_jam else self.r_of[t]
            reach = np.ones(N)
            v_cond = np.zeros(N)
            for q in resp:
                y = ys[(q, t)]
                a_q = self.P * (x[t] * reach)[:, None]
                if is_jam:
                    v_agg[(q, t)] = (a_q * self.call_pay[q]).sum(axis=0)
                    tq = (self.Pcond * self.jam_pay[q]) @ y
                else:
                    z = zs[(t, q)]
                    win = self.resteal_win[(t, q)]
                    v_agg[(q, t)] = (
                        (a_q * (1.0 - z)[:, None]).sum(axis=0) * win
                        + ((a_q * z[:, None]) * self.call_pay[q]).sum(axis=0))
                    # opener's re-jam-response infoset (excludes own x, z)
                    reach_z = self.P * reach[:, None] * y[None, :]
                    v_zcall[(t, q)] = (reach_z * self.jam_pay[q]).sum(axis=1)
                    v_zfold[(t, q)] = reach_z.sum(axis=1) * (-(r + self.ante))
                    tq = ((1.0 - z) * (-(r + self.ante)) * (self.Pcond @ y)
                          + z * ((self.Pcond * self.jam_pay[q]) @ y))
                v_rfold[(q, t)] = a_q.sum(axis=0) * self.resp_fold[q]
                v_cond += reach * tq
                reach = reach * (self.Pcond @ (1.0 - y))
            v_cond += reach * self.steal
            v_open[t] = self.w * v_cond
        v_open["fold"] = self.w * self.fold_ev
        return v_open, v_agg, v_rfold, v_zcall, v_zfold


def _nash_gap(game: _OpenGame, x: dict, ys: dict, zs: dict) -> float:
    """Mean per-player best-response gain, recomputable from strategies.

    The opener's best response optimises the call-vs-re-jam infosets FIRST
    (z* per hand), then picks the root action against those improved
    continuations — per-infoset maxima alone would understate the gain.
    """
    v_open, v_agg, v_rfold, v_zcall, v_zfold = game.values(x, ys, zs)
    zs_star = {k: (v_zcall[k] >= v_zfold[k]).astype(float) for k in v_zcall}
    v_open_star, _, _, _, _ = game.values(x, ys, zs_star)
    labels = game.threats + ["fold"]
    on = sum(x[t] * v_open[t] for t in labels)
    br = np.maximum.reduce([v_open_star[t] for t in labels])
    gain = float((br - on).sum())
    for key, va in v_agg.items():
        q, t = key
        y = ys[key]
        on_q = y * va + (1.0 - y) * v_rfold[key]
        gain += float((np.maximum(va, v_rfold[key]) - on_q).sum())
    return gain / (1 + len(game.responders))


@lru_cache(maxsize=None)
def solve_open(opener: str, depth_bb: float, ante: float = 0.0,
               table: tuple[str, ...] = RING_ORDER,
               sizes: tuple[float, ...] = RAISE_SIZES,
               iters: int = 1500, target_gap: float = 1e-5,
               max_iters: int = 90000) -> OpenSolution:
    """Solve one open-game formation (cached), iterating until certified."""
    _validate_depth(depth_bb)
    _validate_ante(ante)
    _validate_table(table, opener)
    for r in sizes:
        if not (0.0 < float(r) < float(depth_bb)):
            raise ValueError(
                f"raise size {r!r}bb must be positive and below the "
                f"{depth_bb}bb stack — at or above it, the raise IS a jam")
    E = load_equity_matrix().equity_matrix
    P = joint_prior()
    game = _OpenGame(table, opener, float(depth_bb), float(ante),
                     tuple(float(r) for r in sizes), E, P)
    resp = game.responders
    labels = game.threats + ["fold"]
    K = len(labels)

    r_open = np.zeros((K, N))
    s_open = np.zeros((K, N))
    r_y = {(q, t): np.zeros((2, N)) for q in resp for t in game.threats}
    s_y = {k: np.zeros((2, N)) for k in r_y}
    zkeys = [(t, q) for t in game.threats[1:] for q in resp]
    r_z = {k: np.zeros((2, N)) for k in zkeys}
    s_z = {k: np.zeros((2, N)) for k in zkeys}

    t_iter = 0
    while True:
        for _ in range(iters):
            t_iter += 1
            xk = _regret_match_k(r_open)
            x = {lab: xk[j] for j, lab in enumerate(labels)}
            ys = {k: _regret_match(r_y[k][0], r_y[k][1]) for k in r_y}
            zs = {k: _regret_match(r_z[k][0], r_z[k][1]) for k in r_z}
            v_open, v_agg, v_rfold, v_zcall, v_zfold = game.values(x, ys, zs)

            von = sum(x[lab] * v_open[lab] for lab in labels)
            for j, lab in enumerate(labels):
                r_open[j] += v_open[lab] - von
            np.maximum(r_open, 0.0, out=r_open)
            for j, lab in enumerate(labels):
                s_open[j] += t_iter * x[lab]

            for k in r_y:
                vq = ys[k] * v_agg[k] + (1.0 - ys[k]) * v_rfold[k]
                r_y[k][0] += v_agg[k] - vq
                r_y[k][1] += v_rfold[k] - vq
                np.maximum(r_y[k], 0.0, out=r_y[k])
                s_y[k][0] += t_iter * ys[k]
                s_y[k][1] += t_iter * (1.0 - ys[k])
            for k in r_z:
                vz = zs[k] * v_zcall[k] + (1.0 - zs[k]) * v_zfold[k]
                r_z[k][0] += v_zcall[k] - vz
                r_z[k][1] += v_zfold[k] - vz
                np.maximum(r_z[k], 0.0, out=r_z[k])
                s_z[k][0] += t_iter * zs[k]
                s_z[k][1] += t_iter * (1.0 - zs[k])

        tot = s_open.sum(axis=0)
        x = {lab: s_open[j] / tot for j, lab in enumerate(labels)}
        ys = {k: s_y[k][0] / (s_y[k][0] + s_y[k][1]) for k in s_y}
        zs = {k: s_z[k][0] / (s_z[k][0] + s_z[k][1]) for k in s_z}
        if _nash_gap(game, x, ys, zs) <= target_gap or t_iter >= max_iters:
            break

    v_open, v_agg, _, _, _ = game.values(x, ys, zs)
    with np.errstate(invalid="ignore", divide="ignore"):
        open_ev = {lab: np.where(game.w > 0, v_open[lab] / game.w, 0.0)
                   for lab in labels}
        defend_ev = {}
        for t in game.threats:
            reach = np.ones(N)
            for q in resp:
                node = (game.P * (x[t] * reach)[:, None]).sum(axis=0)
                defend_ev[(q, t)] = np.where(
                    node > 0, v_agg[(q, t)] / node, 0.0)
                reach = reach * (game.Pcond @ (1.0 - ys[(q, t)]))

    return OpenSolution(
        table=table, opener=opener, depth_bb=float(depth_bb),
        ante=float(ante), sizes=game.sizes,
        open_freq=x, open_ev=open_ev,
        defend_freq=dict(ys), defend_ev=defend_ev,
        defend_fold_ev={(q, t): game.resp_fold[q]
                        for q in resp for t in game.threats},
        callback_freq=dict(zs),
        exploitability=_nash_gap(game, x, ys, zs),
    )


# --------------------------------------------------------------------------- #
# Checked-in artifact (ring_charts pattern). The open-game grid takes tens of
# minutes to solve, so it ships pre-solved; the cache test re-certifies every
# stored formation's Nash gap from scratch on every fast-suite run. Arrays are
# float32 — plenty for frequencies/EVs against a 0.1bb ε floor, and it halves
# a multi-megabyte artifact.
# --------------------------------------------------------------------------- #
DATA_PATH = Path(__file__).resolve().parent / "data" / "open_charts.npz"

# The drill grid's openers. "SBhu" is the heads-up table; everything else is
# 9-max. SB appears in BOTH: folded-to-SB at a full ring has seven extra
# antes of dead money, which is exactly the seam ring.py documents.
OPEN_FORMATIONS: dict[str, tuple[tuple[str, ...], str]] = {
    **{p: (RING_ORDER, p) for p in
       ("UTG", "UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB")},
    "SBhu": (("SB", "BB"), "SB"),
}


def open_cache_key(formation: str, depth_bb: float, ante: float) -> str:
    return f"{formation}|{depth_bb:g}|{ante:g}"


@lru_cache(maxsize=1)
def load_open_charts(path: str | None = None) -> dict[str, OpenSolution]:
    p = Path(path) if path else DATA_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"open chart artifact missing: {p}\n"
            "generate it with: uv run python scripts/gen_open_charts.py")
    out: dict[str, OpenSolution] = {}
    with np.load(p) as d:
        keys = sorted({name.split("/")[0] for name in d.files})
        for key in keys:
            formation = key.split("|")[0]
            table, opener = OPEN_FORMATIONS[formation]
            behind = table[table.index(opener) + 1:]
            depth, ante, expl = (float(v) for v in d[f"{key}/meta"])
            # float32 storage: 2.2 comes back 2.2000000476…, which would also
            # corrupt the "raise 2.2bb" labels rebuilt from it — round to the
            # grid's actual precision
            sizes = tuple(round(float(r), 6) for r in d[f"{key}/sizes"])
            threats = ["jam"] + [_rlabel(r) for r in sizes]
            labels = threats + ["fold"]
            of = d[f"{key}/open_freq"].astype(np.float64)
            oe = d[f"{key}/open_ev"].astype(np.float64)
            df = d[f"{key}/defend_freq"].astype(np.float64)
            de = d[f"{key}/defend_ev"].astype(np.float64)
            dfe = d[f"{key}/defend_fold_ev"].astype(np.float64)
            zf = d[f"{key}/callback_freq"].astype(np.float64)
            defend_freq, defend_ev, defend_fold = {}, {}, {}
            for ti, t in enumerate(threats):
                for qi, q in enumerate(behind):
                    defend_freq[(q, t)] = df[ti, qi]
                    defend_ev[(q, t)] = de[ti, qi]
                    defend_fold[(q, t)] = float(dfe[ti, qi])
            callback = {}
            for ri, t in enumerate(threats[1:]):
                for qi, q in enumerate(behind):
                    callback[(t, q)] = zf[ri, qi]
            out[key] = OpenSolution(
                table=table, opener=opener, depth_bb=depth, ante=ante,
                sizes=sizes,
                open_freq={lab: of[j] for j, lab in enumerate(labels)},
                open_ev={lab: oe[j] for j, lab in enumerate(labels)},
                defend_freq=defend_freq, defend_ev=defend_ev,
                defend_fold_ev=defend_fold, callback_freq=callback,
                exploitability=expl,
            )
    return out


def open_solution(formation: str, depth_bb: float, ante: float = 0.0
                  ) -> OpenSolution:
    """A formation's solve from the artifact; live solve if absent (same
    solver, same budget — pinned by the slow cache test)."""
    cached = load_open_charts().get(open_cache_key(formation, depth_bb, ante))
    if cached is not None:
        return cached
    table, opener = OPEN_FORMATIONS[formation]
    return solve_open(opener, depth_bb, ante, table=table,
                      sizes=sizes_for_depth(depth_bb))


def open_ctx(sol: OpenSolution, formation: str) -> str:
    sizes = ",".join(f"{r:g}" for r in sol.sizes)
    return (f"chart:open:{formation}:{sol.depth_bb:g}bb:a{sol.ante:g}:"
            f"sizes={sizes}:single-aggressor|no-flat-calls|"
            f"pairwise-removal|stacks-symmetric")
