"""Jam/fold Nash chart engine (impl doc §3 Slice C; plan §5.1).

Models the ≤20bb push/fold spot as a two-player game: SB open-jams or folds;
BB (facing a jam) calls or folds. Card classes are the 169 preflop types; the
payoffs use the checked-in equity matrix + blinds/antes. Solved with a
vectorised CFR+ over the 169 infosets (each 2 actions) — mathematically the
same regret-matching CFR that Slice B runs on a game tree, but expressed as
numpy so 5–20bb solves take milliseconds.

Two payoff models share the same solver:
  * chip-EV  — strictly zero-sum; exploitability of the solved profile → 0.
  * ICM $    — general-sum (the two idle players absorb ICM equity); used for
               bubble push/fold where risk premium tightens calling ranges.

`JamFoldGame` re-expresses a (small) instance as a Slice-B `Game`, so the numpy
math is cross-checked against `cfr.CFRSolver` + `cfr.exploit` on a tiny fixture
before being trusted at 169-hand scale (test_charts_jamfold).

Public API (consumed by Slice E drills):
    jamfold_range(position, depth_bb, ante=0) -> {hand_label: types.Solution}
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from pokerlab.types import Solution

from . import hands
from .equity import EquityMatrix, load_equity_matrix
from .icm import icm_equities

N = 169


# --------------------------------------------------------------------------- #
# Joint prior over (SB class, BB class) with exact card removal.
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def joint_prior() -> np.ndarray:
    """Normalised P[i,j] = P(SB dealt class i, BB dealt class j), card-removal
    aware. joint[i,j] = combos_i·combos_j − Σ_c N_i(c)N_j(c) + δ_ij·combos_i,
    where N_i(c) counts class-i concrete combos using card c."""
    counts = np.zeros((N, 52), dtype=np.float64)
    combo_n = np.zeros(N, dtype=np.float64)
    for i, label in enumerate(hands.HAND_CLASSES):
        cc = hands.card_combos(label)
        combo_n[i] = len(cc)
        for c1, c2 in cc:
            counts[i, c1] += 1
            counts[i, c2] += 1
    joint = np.outer(combo_n, combo_n) - counts @ counts.T
    joint += np.diag(combo_n)
    joint = np.maximum(joint, 0.0)
    return joint / joint.sum()


# --------------------------------------------------------------------------- #
# Payoff models.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class JamFoldModel:
    """Per-player terminal payoffs (bb for chip model, $ delta for ICM)."""

    call_sb: np.ndarray   # (N,N) SB payoff when SB jams & BB calls
    call_bb: np.ndarray   # (N,N) BB payoff, same terminal
    bbfold_sb: float      # SB jams, BB folds
    bbfold_bb: float
    sbfold_sb: float      # SB folds
    sbfold_bb: float
    zero_sum: bool


def chip_model(depth_bb: float, ante: float, E: np.ndarray) -> JamFoldModel:
    """Classic HU push/fold chip-EV payoffs (bb), stacks = depth_bb each.

    SB fold: -(0.5+ante).  SB jam/BB fold: +(1+ante).  Showdown for stack s:
    net (2·equity−1)·s.  Dead money (blinds+antes) is absorbed into s at showdown.
    """
    s = float(depth_bb)
    a = float(ante)
    call_sb = (2.0 * E - 1.0) * s
    return JamFoldModel(
        call_sb=call_sb,
        call_bb=-call_sb,
        bbfold_sb=1.0 + a,
        bbfold_bb=-(1.0 + a),
        sbfold_sb=-(0.5 + a),
        sbfold_bb=0.5 + a,
        zero_sum=True,
    )


# --------------------------------------------------------------------------- #
# Input validation for the public chart entry points (round-2 findings
# [E34][E35][E36]). These solves are answer keys for live drills and for HH
# grading, so a degenerate input must fail loudly rather than converge to a
# confident-looking chart: an empty prize ladder makes every ICM delta zero and
# yields a uniform 50/50 chart whose Nash gap is a genuine 0.0 (it IS an
# equilibrium of a game where nothing matters), and a non-positive depth flips
# the showdown term's sign and converges to the MIRROR of the correct chart —
# folding aces and jamming 32o — with no warning at all.
# --------------------------------------------------------------------------- #
def _validate_depth(depth_bb: float) -> None:
    if not math.isfinite(depth_bb) or depth_bb <= 0.0:
        raise ValueError(
            f"depth_bb must be finite and positive, got {depth_bb!r} — a "
            "non-positive depth inverts the chart rather than failing")


def _validate_payouts(payouts: Sequence[float]) -> None:
    if not len(payouts):
        raise ValueError("ICM needs a non-empty prize ladder")
    vals = [float(p) for p in payouts]
    if not all(math.isfinite(p) and p >= 0.0 for p in vals):
        raise ValueError(f"payouts must be finite and non-negative, got {vals!r}")
    if sum(vals) <= 0.0:
        raise ValueError(
            "prize ladder sums to zero — every ICM delta would be 0 and the "
            "solve would report a meaningless exploitability of 0.0")
    if any(b > a for a, b in zip(vals, vals[1:])):
        raise ValueError(f"payouts must be descending, got {vals!r}")


def _validate_icm_inputs(stacks: Sequence[float], sb_seat: int, bb_seat: int,
                         payouts: Sequence[float], bb_chips: int) -> None:
    _validate_payouts(payouts)
    n = len(stacks)
    if n < 2:
        raise ValueError(f"ICM push/fold needs at least 2 seats, got {n}")
    if not (0 <= sb_seat < n and 0 <= bb_seat < n):
        raise ValueError(f"seat index out of range: sb={sb_seat} bb={bb_seat} n={n}")
    if sb_seat == bb_seat:
        raise ValueError(f"sb_seat and bb_seat must differ (both {sb_seat})")
    if not all(math.isfinite(s) and s >= 0.0 for s in stacks):
        raise ValueError(f"stacks must be finite and non-negative, got {list(stacks)!r}")
    if not (math.isfinite(bb_chips) and bb_chips > 0):
        raise ValueError(f"bb_chips must be finite and positive, got {bb_chips!r}")


def icm_model(
    stacks_chips: list[int],
    sb_seat: int,
    bb_seat: int,
    payouts: list[int],
    bb_chips: int,
    ante: float,
    E: np.ndarray,
) -> JamFoldModel:
    """ICM-$ payoffs for the bubble push/fold spot (general-sum).

    Payoff to each of SB/BB = ICM equity after the outcome minus baseline. The
    two idle seats keep their stacks; because ICM is non-linear, the two active
    players' $ swings do NOT cancel — that asymmetry is the risk premium that
    tightens ranges near the bubble.
    """
    stacks = [float(x) for x in stacks_chips]
    _validate_icm_inputs(stacks, sb_seat, bb_seat, payouts, bb_chips)
    base = icm_equities(stacks, payouts)
    eff = min(stacks[sb_seat], stacks[bb_seat])  # matched all-in size (chips)
    a_ch = ante * bb_chips
    sb_blind = 0.5 * bb_chips
    # A seat cannot post more than it has: short of a full blind it is simply
    # all-in for its stack. Charging the nominal blind regardless drove stacks
    # negative and produced negative ICM equity (round-2 finding [E36]) — the
    # forced all-in IS the honest game here, so clamp rather than reject.
    sb_post = min(sb_blind + a_ch, stacks[sb_seat])
    bb_post = min(bb_chips + a_ch, stacks[bb_seat])

    def delta(sb_delta: float, bb_delta: float) -> tuple[float, float]:
        v = list(stacks)
        v[sb_seat] += sb_delta
        v[bb_seat] += bb_delta
        eq = icm_equities(v, payouts)
        return eq[sb_seat] - base[sb_seat], eq[bb_seat] - base[bb_seat]

    sbfold = delta(-sb_post, sb_post)
    bbfold = delta(bb_post, -bb_post)
    win = delta(eff, -eff)     # SB wins the all-in
    lose = delta(-eff, eff)    # SB loses the all-in

    call_sb = E * win[0] + (1.0 - E) * lose[0]
    call_bb = E * win[1] + (1.0 - E) * lose[1]
    return JamFoldModel(
        call_sb=call_sb,
        call_bb=call_bb,
        bbfold_sb=bbfold[0],
        bbfold_bb=bbfold[1],
        sbfold_sb=sbfold[0],
        sbfold_bb=sbfold[1],
        zero_sum=False,
    )


# --------------------------------------------------------------------------- #
# Counterfactual value helpers (shared by solver + exploitability).
# --------------------------------------------------------------------------- #
def _bb_values(model: JamFoldModel, P: np.ndarray, x: np.ndarray):
    """BB infoset action values (chance+SB reach weighted) given SB jam probs x."""
    reach = P * x[:, None]                     # reach[i,j] = P[i,j]·x_i
    v_call = (reach * model.call_bb).sum(axis=0)
    v_fold = reach.sum(axis=0) * model.bbfold_bb
    return v_call, v_fold


def _sb_values(model: JamFoldModel, P: np.ndarray, y: np.ndarray, w: np.ndarray):
    """SB infoset action values (chance reach weighted) given BB call probs y."""
    jam = P * (y[None, :] * model.call_sb + (1.0 - y)[None, :] * model.bbfold_sb)
    v_jam = jam.sum(axis=1)
    v_fold = w * model.sbfold_sb
    return v_jam, v_fold


def _regret_match(reg0: np.ndarray, reg1: np.ndarray) -> np.ndarray:
    """Prob of action 0 (jam/call) from a 2-action positive-regret pair."""
    p0 = np.maximum(reg0, 0.0)
    p1 = np.maximum(reg1, 0.0)
    tot = p0 + p1
    return np.where(tot > 0.0, p0 / np.where(tot > 0.0, tot, 1.0), 0.5)


# --------------------------------------------------------------------------- #
# Solution container + solver.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class JamFoldSolution:
    depth_bb: float
    ante: float
    model_name: str
    sb_jam: np.ndarray      # (N,) SB jam frequency per class
    bb_call: np.ndarray     # (N,) BB call frequency per class
    sb_jam_ev: np.ndarray   # (N,) per-hand EV of jamming
    sb_fold_ev: float
    bb_call_ev: np.ndarray  # (N,) per-hand EV of calling a jam
    bb_fold_ev: float
    exploitability: float

    def sb_jam_combos(self) -> float:
        return float(sum(self.sb_jam[i] * hands.combos(h)
                         for i, h in enumerate(hands.HAND_CLASSES)))

    def bb_call_combos(self) -> float:
        return float(sum(self.bb_call[i] * hands.combos(h)
                         for i, h in enumerate(hands.HAND_CLASSES)))


def solve_model(model: JamFoldModel, P: np.ndarray, iters: int = 1500) -> tuple:
    """Vectorised CFR+ (simultaneous updates, linear averaging). Returns
    (x_avg, y_avg, w) — SB jam / BB call frequencies + SB marginal reach."""
    n = P.shape[0]
    w = P.sum(axis=1)
    Rsb = np.zeros((2, n))
    Rbb = np.zeros((2, n))
    Ssb = np.zeros((2, n))
    Sbb = np.zeros((2, n))
    for t in range(1, iters + 1):
        x = _regret_match(Rsb[0], Rsb[1])
        y = _regret_match(Rbb[0], Rbb[1])

        v_call, v_fold_bb = _bb_values(model, P, x)
        v_bb = y * v_call + (1.0 - y) * v_fold_bb
        Rbb[0] += v_call - v_bb
        Rbb[1] += v_fold_bb - v_bb

        v_jam, v_fold_sb = _sb_values(model, P, y, w)
        v_sb = x * v_jam + (1.0 - x) * v_fold_sb
        Rsb[0] += v_jam - v_sb
        Rsb[1] += v_fold_sb - v_sb

        np.maximum(Rsb, 0.0, out=Rsb)
        np.maximum(Rbb, 0.0, out=Rbb)

        Ssb[0] += t * x
        Ssb[1] += t * (1.0 - x)
        Sbb[0] += t * y
        Sbb[1] += t * (1.0 - y)

    x_avg = Ssb[0] / (Ssb[0] + Ssb[1])
    y_avg = Sbb[0] / (Sbb[0] + Sbb[1])
    return x_avg, y_avg, w


def model_exploitability(model: JamFoldModel, P: np.ndarray,
                         x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Mean per-player best-response gap. Gain = BR value − on-policy.

    For the zero-sum chip model this is nash_conv/2 in bb. For the general-sum
    ICM model there is no nash_conv, but the same quantity is still the Nash
    gap (a profile is an equilibrium exactly when no player gains by deviating)
    — there it reads in ICM $ (finding [20]).
    """
    v_call, v_fold_bb = _bb_values(model, P, x)
    v_bb = y * v_call + (1.0 - y) * v_fold_bb
    gain_bb = float((np.maximum(v_call, v_fold_bb) - v_bb).sum())

    v_jam, v_fold_sb = _sb_values(model, P, y, w)
    v_sb = x * v_jam + (1.0 - x) * v_fold_sb
    gain_sb = float((np.maximum(v_jam, v_fold_sb) - v_sb).sum())
    return (gain_sb + gain_bb) / 2.0


@lru_cache(maxsize=None)
def solve_jamfold(depth_bb: float, ante: float = 0.0,
                  iters: int = 1500) -> JamFoldSolution:
    """Solve the chip-EV HU push/fold game at a given depth (cached)."""
    _validate_depth(depth_bb)
    em = load_equity_matrix()
    E = em.equity_matrix
    P = joint_prior()
    model = chip_model(depth_bb, ante, E)
    x, y, w = solve_model(model, P, iters)
    expl = model_exploitability(model, P, x, y, w)
    v_jam, _ = _sb_values(model, P, y, w)
    v_call, _ = _bb_values(model, P, x)
    reach_bb = (P * x[:, None]).sum(axis=0)   # BB infoset reach (SB jams into it)
    with np.errstate(invalid="ignore", divide="ignore"):
        jam_ev = np.where(w > 0, v_jam / w, 0.0)
        call_ev = np.where(reach_bb > 0, v_call / reach_bb, 0.0)
    return JamFoldSolution(
        depth_bb=float(depth_bb), ante=float(ante), model_name="chip",
        sb_jam=x, bb_call=y, sb_jam_ev=jam_ev, sb_fold_ev=model.sbfold_sb,
        bb_call_ev=call_ev, bb_fold_ev=model.bbfold_bb, exploitability=expl,
    )


def solve_jamfold_icm(
    stacks_chips: tuple[int, ...],
    sb_seat: int,
    bb_seat: int,
    payouts: tuple[int, ...],
    bb_chips: int,
    ante: float = 0.0,
    iters: int = 1500,
) -> JamFoldSolution:
    """Solve the ICM-$ push/fold game (general-sum) on a bubble fixture."""
    em = load_equity_matrix()
    E = em.equity_matrix
    P = joint_prior()
    model = icm_model(list(stacks_chips), sb_seat, bb_seat,
                      list(payouts), bb_chips, ante, E)
    x, y, w = solve_model(model, P, iters)
    depth = min(stacks_chips[sb_seat], stacks_chips[bb_seat]) / bb_chips
    # The ICM game is general-sum, so there is no zero-sum NashConv — but the
    # per-player best-response gap is still exactly the Nash gap, and it is what
    # has to be small for these ranges to serve as answer keys for the LIVE M2
    # bubble drills. Reporting NaN left the solve unverified (finding [20]).
    #
    # Reported as a FRACTION OF THE PRIZE POOL, not raw ICM $ (round-2 finding
    # [E37]): the raw gap carries the ladder's units, so a fixed threshold on it
    # is really a threshold on how large the payouts happen to be. The checked-in
    # $5/$3/$2 fixture passed a <1e-4 bar at 2.6e-5 while the identical solve on
    # a realistic cents ladder measured 2.6e-3 and failed it. Dividing by the
    # pool makes the number dimensionless and the guard scale-invariant.
    pool = float(sum(payouts))
    expl = model_exploitability(model, P, x, y, w) / pool
    return JamFoldSolution(
        depth_bb=float(depth), ante=float(ante), model_name="icm",
        sb_jam=x, bb_call=y, sb_jam_ev=np.zeros(N), sb_fold_ev=model.sbfold_sb,
        bb_call_ev=np.zeros(N), bb_fold_ev=model.bbfold_bb,
        exploitability=expl,
    )


# --------------------------------------------------------------------------- #
# Game-tree re-expression — cross-checks the numpy math against Slice-B CFR.
# --------------------------------------------------------------------------- #
def restrict_model(model: JamFoldModel, idx: list[int]) -> JamFoldModel:
    ix = np.ix_(idx, idx)
    return JamFoldModel(
        call_sb=model.call_sb[ix], call_bb=model.call_bb[ix],
        bbfold_sb=model.bbfold_sb, bbfold_bb=model.bbfold_bb,
        sbfold_sb=model.sbfold_sb, sbfold_bb=model.sbfold_bb,
        zero_sum=model.zero_sum)


@dataclass(frozen=True)
class _JFState:
    i: int = -1              # SB class index (subset-local); -1 = undealt
    j: int = -1              # BB class index (subset-local)
    hist: tuple = ()         # ("jam",), ("fold",), ("jam","call"), ("jam","fold")


_TERMINALS = {("fold",), ("jam", "call"), ("jam", "fold")}


class JamFoldGame:
    """A (small) jam/fold instance as a Slice-B `Game` (game.py protocol).

    Used to validate the numpy solver: `cfr.CFRSolver` + `cfr.exploit` on this
    tree must reproduce solve_model / model_exploitability on the same instance.
    Indices are subset-local into `model`'s (already restricted) payoff arrays.
    """

    num_players = 2

    def __init__(self, model: JamFoldModel, prior: np.ndarray):
        self.model = model
        self.prior = prior / prior.sum()
        self.n = prior.shape[0]

    def initial_states(self):
        return [(_JFState(), 1.0)]

    def current_player(self, s: _JFState) -> int:
        if s.i < 0:
            return -1  # CHANCE
        if s.hist in _TERMINALS:
            return -4  # TERMINAL
        return 0 if len(s.hist) == 0 else 1

    def chance_outcomes(self, s: _JFState):
        return [((i, j), float(self.prior[i, j]))
                for i in range(self.n) for j in range(self.n)]

    def legal_actions(self, s: _JFState):
        return ["jam", "fold"] if len(s.hist) == 0 else ["call", "fold"]

    def infoset_key(self, s: _JFState, player: int) -> str:
        return f"SB:{s.i}" if player == 0 else f"BB:{s.j}"

    def apply(self, s: _JFState, action):
        if s.i < 0:
            return _JFState(i=action[0], j=action[1], hist=())
        return _JFState(i=s.i, j=s.j, hist=s.hist + (action,))

    def is_terminal(self, s: _JFState) -> bool:
        return s.hist in _TERMINALS

    def returns(self, s: _JFState):
        m = self.model
        if s.hist == ("fold",):
            return [m.sbfold_sb, m.sbfold_bb]
        if s.hist == ("jam", "fold"):
            return [m.bbfold_sb, m.bbfold_bb]
        return [float(m.call_sb[s.i, s.j]), float(m.call_bb[s.i, s.j])]


# --------------------------------------------------------------------------- #
# Public API for drills (Slice E).
# --------------------------------------------------------------------------- #
def jamfold_range(position: str, depth_bb: float, ante: float = 0.0
                  ) -> dict[str, Solution]:
    """Per-hand `types.Solution` for the SB jammer or BB caller at a depth.

    position "SB": actions {"jam","fold"}.  position "BB": {"call","fold"}.
    Frequencies come from the chip-EV Nash solve; source="chart".
    """
    sol = solve_jamfold(float(depth_bb), float(ante))
    ctx = f"chart:jamfold:{position.upper()}:{depth_bb}bb:ante{ante}"
    out: dict[str, Solution] = {}
    if position.upper() == "SB":
        for i, label in enumerate(hands.HAND_CLASSES):
            f = float(sol.sb_jam[i])
            out[label] = Solution(
                actions={"jam": (float(sol.sb_jam_ev[i]), f),
                         "fold": (float(sol.sb_fold_ev), 1.0 - f)},
                range_ctx=ctx, source="chart")
    elif position.upper() == "BB":
        for i, label in enumerate(hands.HAND_CLASSES):
            f = float(sol.bb_call[i])
            out[label] = Solution(
                actions={"call": (float(sol.bb_call_ev[i]), f),
                         "fold": (float(sol.bb_fold_ev), 1.0 - f)},
                range_ctx=ctx, source="chart")
    else:
        raise ValueError(f"position must be 'SB' or 'BB', got {position!r}")
    return out
