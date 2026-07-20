"""Tiny-HUNL depth-limited value-net spike (Slice I item 1; plan M7 / §7 L4).

The M5 ReBeL pattern (rebel.trunk) generalized to HUNL turn/river, reusing the
Slice-D subgame solver as the exact oracle AND the data-gen labeller:

  1. DATA-GEN — sample river subgames (5-card boards, 20bb, varied off-blueprint
     beliefs), solve each exactly (Slice-D CFR+), and record a PBS -> CFV row:
     features = board + pot + both players' per-class belief; target = the OOP
     per-class root counterfactual value. Parquet via pyarrow. Boards are drawn
     i.i.d. and their flop iso-class is RECORDED per row — the rows are not
     stratified across iso-classes, which the docs used to claim (wave-3 [M9]).
  2. VALUE NET — a small MLP PBS -> per-class CFV, trained locally in seconds.
  3. EVAL (the L4 go/no-go) — on >=20 HELD-OUT *turn* subgames, compare the
     net-driven depth-limited turn solver to the exact oracle by mean
     exploitability, measured with Slice-D's verified best response.

The depth-limited solver (`DepthLimitedTurnSolver`) is `SubgameSolver` with the
river chance node replaced by a net leaf: it averages the net's river CFV over
the runout. Exploitability is then measured in the FULL turn+river game
(net turn strategy + exact river continuation) via the oracle's BR — so the
number reflects the net's approximation error, exactly per plan §7 L4.

Encodes the public line/history in the features (Slice-G note) via the board +
pot + belief PBS. No cloud spend: everything is local numpy/torch minutes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pokerlab.charts import hands
from pokerlab.solver import subgame as sg
from pokerlab.solver.iso import iso_class
from pokerlab.solver.subgame import COMBO_INDEX, NUM_COMBOS, Chance, SubgameSolver

BB = 100                       # chips per big blind (unit)
STACK_BB = 20.0                # tiny-HUNL effective stack
# 2 bet sizes + jam. max_raises=0 keeps the tree shallow (no re-raise wars) so
# the turn+river oracle stays a local-minutes solve — the tiny-HUNL scope.
TINY_CFG = sg.BetConfig(sizes=(0.5, 1.0), jam=True, max_raises=0)
N_CLASSES = len(hands.HAND_CLASSES)  # 169

# combo id (0..1325) -> 169-class index, precomputed once.
_COMBO_TO_CLASS = np.full(NUM_COMBOS, -1, dtype=np.int64)
for _ci, _label in enumerate(hands.HAND_CLASSES):
    for _a, _b in hands.card_combos(_label):
        _COMBO_TO_CLASS[COMBO_INDEX[(min(_a, _b), max(_a, _b))]] = _ci
assert (_COMBO_TO_CLASS >= 0).all()


# --------------------------------------------------------------------------- #
# Belief / feature encoding (the PBS).
# --------------------------------------------------------------------------- #
def class_reach(range_combos: np.ndarray) -> np.ndarray:
    """Aggregate a 1326-combo reach vector to per-class total reach (169)."""
    out = np.zeros(N_CLASSES)
    np.add.at(out, _COMBO_TO_CLASS, range_combos)
    return out


def features(board: tuple[int, ...], pot: float, stack: float,
             r0: np.ndarray, r1: np.ndarray) -> np.ndarray:
    """PBS feature vector: board multi-hot + pot + stack behind + both beliefs.

    ``stack`` (chips behind per player) is part of the public state and is NOT
    implied by the pot: a river subgame with 20bb behind is a different game
    from the same pot with 0 behind. Omitting it made the depth-limited leaf
    unrepresentable, since a river entry after turn betting always has less
    behind than the root (round-2 follow-up to finding [19]).
    """
    board_hot = np.zeros(52)
    board_hot[list(board)] = 1.0
    c0, c1 = class_reach(r0), class_reach(r1)
    c0 = c0 / (c0.sum() or 1.0)
    c1 = c1 / (c1.sum() or 1.0)
    return np.concatenate([board_hot, [pot / STACK_BB, stack / STACK_BB],
                           c0, c1]).astype(np.float32)


FEAT_DIM = 52 + 2 + N_CLASSES + N_CLASSES
# Two heads: OOP per-class CFV, then IP per-class CFV. IP is NOT the zero-sum
# complement of OOP — the subgame carries dead money (per matchup the two
# payoffs sum to pot0, not 0) and the two players index different hands — so
# each player gets its own target (round-1 finding [19]).
OUT_DIM = 2 * N_CLASSES


# --------------------------------------------------------------------------- #
# Range sampling (off-blueprint beliefs).
# --------------------------------------------------------------------------- #
def _range_from_class_weights(w: np.ndarray, board: tuple[int, ...]) -> np.ndarray:
    r = w[_COMBO_TO_CLASS].astype(np.float64)
    return r * sg.board_mask(board)


# Range density used to GENERATE training data. Evaluation must draw from the
# same distribution or the net is queried off its own training support, so both
# sides read this one constant rather than carrying separate literals that drift
# (wave-3 [M2]).
GEN_KEEP_FRAC = 0.2


def sample_range(rng: np.random.Generator, board: tuple[int, ...],
                 *, keep_frac: float = GEN_KEEP_FRAC) -> np.ndarray:
    """A varied, off-blueprint, SPARSE range: a random subset of classes gets
    weight (realistic + keeps the live-combo count — and thus solve cost — low),
    occasionally strength-tilted, always board-masked."""
    w = rng.random(N_CLASSES)
    if rng.random() < 0.5:  # tilt toward a random strength band sometimes
        w = w ** rng.uniform(0.5, 3.0)
    w[rng.random(N_CLASSES) > keep_frac] = 0.0  # sparsify the range
    if not w.any():
        w[rng.integers(N_CLASSES)] = 1.0
    return _range_from_class_weights(w, board)


def _deal_board(rng: np.random.Generator, n_cards: int) -> tuple[int, ...]:
    return tuple(int(c) for c in rng.choice(52, size=n_cards, replace=False))


# --------------------------------------------------------------------------- #
# Oracle CFV labels from the Slice-D solver.
# --------------------------------------------------------------------------- #
def solve_river(board: tuple[int, ...], pot0: float, r0: np.ndarray, r1: np.ndarray,
                *, iters: int, stack: float = STACK_BB,
                cfg: sg.BetConfig = TINY_CFG) -> SubgameSolver:
    tree = sg.build_tree(board, pot0=pot0, stack=stack, cfg=cfg)
    s = SubgameSolver(tree, board, r0.copy(), r1.copy(), pot0=pot0)
    s.iterate(iters)
    return s


def river_entry_states(pot0: float, *, stack: float = STACK_BB,
                       cfg: sg.BetConfig = TINY_CFG) -> list[tuple[float, float]]:
    """The (pot, stack-behind) states a turn subgame can reach at a river deal.

    This is exactly the distribution the depth-limited leaf gets queried at, so
    data-gen samples from it rather than assuming the turn root's pot and a full
    stack. Derived from the tree itself, so it tracks the bet grid automatically.
    """
    tree = sg.build_tree((0, 1, 2, 3), pot0=pot0, stack=stack, cfg=cfg)
    out: set[tuple[float, float]] = set()

    def walk(node):
        if isinstance(node, Chance):
            out.add((round(node.pot, 6), round(node.stack, 6)))
            return
        if isinstance(node, sg.Decision):
            for c in node.children:
                walk(c)

    walk(tree)
    return sorted(out)


def _root_normalized_evs(solver: SubgameSolver) -> tuple[np.ndarray, np.ndarray]:
    """Both players' NORMALIZED per-hand root EV, from ONE tree walk.

    Normalized (per-matchup bb) rather than counterfactual because the PBS
    features carry a *normalized* belief — a target that scaled with the
    opponent's absolute reach mass would not be a function of the features.
    `counterfactual_from_normalized` converts back at the depth limit.
    """
    avg = solver._avg_compressed()
    v0, v1 = solver._walk(solver.root, solver.board,
                          solver._r0.copy(), solver._r1.copy(), avg=avg)
    return _normalize(solver, v0, v1, solver._r0, solver._r1)


def _by_class(solver: SubgameSolver, per_combo_v: np.ndarray,
              my_reach: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reach-weighted average within each of the 169 classes, + presence mask."""
    classes = _COMBO_TO_CLASS[solver.live]
    num = np.zeros(N_CLASSES)
    den = np.zeros(N_CLASSES)
    np.add.at(num, classes, per_combo_v * my_reach)
    np.add.at(den, classes, my_reach)
    cfv = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    return cfv.astype(np.float32), (den > 0).astype(np.float32)


def class_cfv(solver: SubgameSolver, player: int) -> tuple[np.ndarray, np.ndarray]:
    """``player``'s per-class normalized root value + per-class presence mask."""
    evs = _root_normalized_evs(solver)
    return _by_class(solver, evs[player],
                     solver._r0 if player == 0 else solver._r1)


def oop_class_cfv(solver: SubgameSolver) -> tuple[np.ndarray, np.ndarray]:
    """OOP per-class root value + mask (the head-0 target). See `class_cfv`."""
    return class_cfv(solver, 0)


def both_class_cfv(solver: SubgameSolver) -> tuple[np.ndarray, np.ndarray]:
    """The stacked two-head training target: [OOP 169 | IP 169] + its mask.

    Shares a single tree walk between the heads — labelling is the dominant
    cost of data-gen, and the two targets differ only in the aggregation.
    """
    ev0, ev1 = _root_normalized_evs(solver)
    c0, m0 = _by_class(solver, ev0, solver._r0)
    c1, m1 = _by_class(solver, ev1, solver._r1)
    return np.concatenate([c0, c1]), np.concatenate([m0, m1])


# --------------------------------------------------------------------------- #
# Leaf units: normalized per-hand EV <-> opponent-reach-weighted counterfactual.
#
# `SubgameSolver._walk` propagates COUNTERFACTUAL values — see
# `_terminal_values`, where every payoff is multiplied by the opponent's
# valid-reach mass (`valid_reach`, card-removal exact). The value net is trained
# on normalized per-hand EV, so its output MUST be converted at the depth limit
# or the leaf branch enters CFR one-to-two orders of magnitude light and the
# turn solve simply ignores it (round-1 finding [19]).
# --------------------------------------------------------------------------- #
def _normalize(solver: SubgameSolver, v0: np.ndarray, v1: np.ndarray,
               reach0: np.ndarray, reach1: np.ndarray):
    valid0 = sg.valid_reach(reach1, solver.c1, solver.c2)  # OOP's opponent = IP
    valid1 = sg.valid_reach(reach0, solver.c1, solver.c2)
    ev0 = np.where(valid0 > 0, v0 / np.where(valid0 > 0, valid0, 1.0), 0.0)
    ev1 = np.where(valid1 > 0, v1 / np.where(valid1 > 0, valid1, 1.0), 0.0)
    return ev0, ev1


def normalized_from_counterfactual(solver: SubgameSolver, v0, v1, reach0, reach1):
    """Counterfactual values -> normalized per-hand EV, at this node's belief."""
    return _normalize(solver, v0, v1, reach0, reach1)


def counterfactual_from_normalized(solver: SubgameSolver, ev0, ev1, reach0, reach1):
    """Normalized per-hand EV -> the counterfactual units `_walk` propagates."""
    return (ev0 * sg.valid_reach(reach1, solver.c1, solver.c2),
            ev1 * sg.valid_reach(reach0, solver.c1, solver.c2))


# --------------------------------------------------------------------------- #
# Dataset generation (iso-class RECORDED per row, not stratified) + Parquet.
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    feats: np.ndarray
    cfv: np.ndarray
    mask: np.ndarray
    iso: str
    pot: float
    stack: float


def generate_samples(n: int, *, seed: int, iters: int = 120) -> list[Sample]:
    """River PBS -> CFV rows, sampled at the states the leaf is queried at.

    A turn root pot is drawn, then one of the (pot, stack) states that turn
    subgame can actually reach at its river deal — so the net is trained on the
    distribution it will be asked about, rather than always at the root pot with
    a full stack (round-2 follow-up to finding [19]).
    """
    rng = np.random.default_rng(seed)
    out: list[Sample] = []
    while len(out) < n:
        board = _deal_board(rng, 5)
        r0 = sample_range(rng, board)
        r1 = sample_range(rng, board)
        if r0.sum() == 0 or r1.sum() == 0:
            continue
        turn_pot0 = float(rng.uniform(2.0, STACK_BB))
        states = river_entry_states(turn_pot0)
        pot, stack = states[int(rng.integers(len(states)))]
        s = solve_river(board, pot, r0, r1, iters=iters, stack=stack)
        cfv, mask = both_class_cfv(s)
        out.append(Sample(features(board, pot, stack, r0, r1), cfv, mask,
                          iso_class(tuple(board[:3])), pot, stack))
    return out


def write_parquet(samples: list[Sample], path: str | Path) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({
        "features": [s.feats.tolist() for s in samples],
        "cfv": [s.cfv.tolist() for s in samples],
        "mask": [s.mask.tolist() for s in samples],
        "iso": [s.iso for s in samples],
    })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def read_parquet(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq

    t = pq.read_table(path)
    X = np.array(t.column("features").to_pylist(), dtype=np.float32)
    Y = np.array(t.column("cfv").to_pylist(), dtype=np.float32)
    M = np.array(t.column("mask").to_pylist(), dtype=np.float32)
    return X, Y, M


# --------------------------------------------------------------------------- #
# Value net (PBS -> per-class CFV).
# --------------------------------------------------------------------------- #
def train_valuenet(X: np.ndarray, Y: np.ndarray, M: np.ndarray, *,
                   epochs: int = 200, hidden: int = 128, seed: int = 0,
                   val_frac: float = 0.2):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    n = len(X)
    n_val = max(1, int(n * val_frac))
    Xt = torch.tensor(X); Yt = torch.tensor(Y); Mt = torch.tensor(M)
    tr = slice(n_val, n); va = slice(0, n_val)

    net = nn.Sequential(
        nn.Linear(FEAT_DIM, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, OUT_DIM),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    def masked_mse(pred, y, m):
        return (((pred - y) ** 2) * m).sum() / m.sum().clamp_min(1.0)

    for _ in range(epochs):
        net.train(); opt.zero_grad()
        loss = masked_mse(net(Xt[tr]), Yt[tr], Mt[tr])
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        val_loss = float(masked_mse(net(Xt[va]), Yt[va], Mt[va]))
    return net, val_loss


def net_class_cfv(net, board: tuple[int, ...], pot: float,
                  r0: np.ndarray, r1: np.ndarray,
                  stack: float = STACK_BB) -> np.ndarray:
    import torch

    x = torch.tensor(features(board, pot, stack, r0, r1)).unsqueeze(0)
    with torch.no_grad():
        return net(x).squeeze(0).numpy()


def net_class_cfv_batch(net, feats: np.ndarray) -> np.ndarray:
    """Batched net inference for a stack of PBS feature rows -> [B, 169]."""
    import torch

    with torch.no_grad():
        return net(torch.tensor(feats)).numpy()


# --------------------------------------------------------------------------- #
# Depth-limited turn solver: SubgameSolver with a net leaf at the river.
# --------------------------------------------------------------------------- #
class DepthLimitedTurnSolver(SubgameSolver):
    """CFR+ over the turn only; the river chance node is replaced by a net leaf.

    ``leaf_fn(board4, reach0_live, reach1_live) -> (v0, v1)`` over the live
    combos. Only ``_walk`` is overridden (training); best response / on-policy
    values are measured on the FULL tree via a plain SubgameSolver.

    UNITS (round-1 finding [19]): ``leaf_fn`` must return values in the same
    opponent-reach-weighted counterfactual units ``_terminal_values`` produces —
    build it with `net_leaf_fn`, which converts the net's normalized per-hand EV
    via `counterfactual_from_normalized`. Returning the net's raw output makes
    the leaf branch ~40-60x too small and CFR silently ignores it.
    """

    def set_leaf(self, leaf_fn):
        self._leaf_fn = leaf_fn
        return self

    def _walk(self, node, board, reach0, reach1, avg=None):
        if isinstance(node, Chance) and len(board) == 4:  # river deal -> net leaf
            # the node's OWN public state, not the subgame root's: turn betting
            # moves both (round-2 follow-up to finding [19])
            return self._leaf_fn(board, reach0, reach1, node.pot, node.stack)
        return super()._walk(node, board, reach0, reach1, avg)


# --------------------------------------------------------------------------- #
# Eval harness — the L4 go/no-go.
# --------------------------------------------------------------------------- #
def _combined_avg(oracle: SubgameSolver, dls: DepthLimitedTurnSolver,
                  turn_nids: set[int]):
    o = oracle._avg_compressed()
    d = dls._avg_compressed()
    return [d[i] if i in turn_nids else o[i] for i in range(len(o))]


def _turn_decision_nids(solver: SubgameSolver) -> set[int]:
    """Decision node ids reached before any river card is dealt (board len 4)."""
    turn: set[int] = set()

    def walk(node, blen):
        from pokerlab.solver.subgame import Decision as _D
        if isinstance(node, _D):
            if blen == 4:
                turn.add(node.nid)
            for c in node.children:
                walk(c, blen)
        elif isinstance(node, Chance):
            for _, c in node.children:
                walk(c, blen + 1)

    walk(solver.root, len(solver.board))
    return turn


def net_leaf_fn(net, dls: "DepthLimitedTurnSolver"):
    """Build the river-entry leaf value function for a depth-limited turn solve.

    Mirrors exactly what the river `Chance` node it replaces would do (see
    `SubgameSolver._walk`): for each possible river card, query the net for both
    players' normalized per-class CFV at the *post-river* belief, convert to
    counterfactual units against that river's blocked reach, mask the combos the
    river kills, and divide by the chance node's divisor.

    ``pot``/``stack`` come from the node being replaced, not the subgame root —
    turn betting moves both, and data-gen samples over the same states.
    """
    live = dls.live
    live_classes = _COMBO_TO_CLASS[live]

    def leaf_fn(board, reach0, reach1, pot, stack):
        used = set(board)
        rivers = [c for c in range(52) if c not in used]
        divisor = float(52 - len(board) - 4)
        masks = [dls._cmask(c) for c in rivers]
        # belief at the river excludes combos containing the river card — the
        # same masking the training rows were generated under (5-card boards).
        feats = np.stack([
            features(board + (c,), pot, stack, _expand(reach0 * m, live),
                     _expand(reach1 * m, live))
            for c, m in zip(rivers, masks)
        ])
        cls = net_class_cfv_batch(net, feats)              # [R, 2*169]
        ev0 = cls[:, live_classes]                          # [R, L] OOP head
        ev1 = cls[:, N_CLASSES + live_classes]              # [R, L] IP head
        v0 = np.zeros(live.size)
        v1 = np.zeros(live.size)
        for i, m in enumerate(masks):
            c0, c1 = counterfactual_from_normalized(
                dls, ev0[i], ev1[i], reach0 * m, reach1 * m)
            v0 += c0 * m
            v1 += c1 * m
        return v0 / divisor, v1 / divisor

    return leaf_fn


def evaluate_spot(net, board4: tuple[int, ...], pot0: float,
                  r0: np.ndarray, r1: np.ndarray, *, iters: int) -> dict:
    """Exploitability of the net-driven turn strategy vs the exact oracle."""
    oracle = solve_river(board4, pot0, r0, r1, iters=iters)  # full turn+river
    turn_nids = _turn_decision_nids(oracle)

    tree = sg.build_tree(board4, pot0=pot0, stack=STACK_BB, cfg=TINY_CFG)
    dls = DepthLimitedTurnSolver(tree, board4, r0.copy(), r1.copy(), pot0=pot0)

    dls.set_leaf(net_leaf_fn(net, dls)).iterate(iters)
    combined = _combined_avg(oracle, dls, turn_nids)
    return {
        "expl_oracle_bb": oracle.exploitability(),
        "expl_net_bb": oracle.exploitability(combined),
        "pot_bb": pot0,
    }


def _expand(reach_live: np.ndarray, live: np.ndarray) -> np.ndarray:
    """Live-subset reach -> full 1326 reach (for feature/belief encoding)."""
    full = np.zeros(NUM_COMBOS)
    full[live] = reach_live
    return full


@dataclass
class SpikeVerdict:
    n_rows: int
    val_loss: float
    n_eval: int
    mean_expl_oracle_bb: float
    mean_expl_net_bb: float
    tolerance: float
    go: bool

    @property
    def ratio(self) -> float:
        """Net exploitability as a MULTIPLE of the depth-limited oracle's own.

        The scale-free number, and the one to quote: it says how much worse the
        learned leaf is than the best this decomposition could do, rather than
        how big it is in bb — which depends on the pot sizes the eval happened
        to draw (wave-3 [M3]/[M6]).
        """
        return float("inf") if self.mean_expl_oracle_bb <= 0 else (
            self.mean_expl_net_bb / self.mean_expl_oracle_bb)


def run_spike(*, n_rows: int = 10_000, n_eval: int = 20, seed: int = 0,
              gen_iters: int = 120, eval_iters: int = 200,
              tolerance: float = 2.0, epochs: int = 300,
              eval_keep_frac: float = GEN_KEEP_FRAC) -> SpikeVerdict:
    """Full local pipeline -> L4 go/no-go verdict (no cloud, minutes).

    THE GO RULE IS RELATIVE (wave-3 [M3]). It used to be
    ``mean_net <= 0.75`` — an absolute bb threshold with no derivation behind
    it, compared against nothing. The baseline it should have been compared to
    was already being computed in the same loop: ``mean_expl_oracle_bb``, the
    exploitability of the SAME depth-limited decomposition with a perfect
    (solved) leaf. That is the floor the learned leaf is trying to reach, so the
    question is "how much worse than the best this design can do", not "how many
    big blinds".

    ``tolerance`` is the multiple of that floor we are willing to accept. It is
    a stated policy choice rather than a derived constant — but the quantity it
    applies to is now derived, which is the part that was wrong. Report
    ``verdict.ratio`` alongside any verdict: it is scale-free, whereas the bb
    figure moves with whatever pot sizes the eval happened to draw.

    ``eval_keep_frac`` defaults to `GEN_KEEP_FRAC` — the SAME density the
    training data was generated at. It used to be sparser (0.12 vs 0.2), which
    put 28.5% of evaluation beliefs below the sparsest belief the net had ever
    seen: the net was being asked about board states outside its own training
    support and then blamed for the answer (wave-3 [M2]).

    Aligning the densities closes that half of the gap. It does NOT close the
    other half: within a solve, beliefs at the leaf are STRATEGY-WEIGHTED
    reaches produced by CFR iterations, while training beliefs are sampled
    independently. No value of this parameter makes an independently-sampled
    range look like a reach vector that a solver walked to. Closing that
    properly means generating training data along the solver's own trajectory —
    the ReBeL self-play loop — which is out of scope for a spike and is recorded
    as an open cause rather than papered over.
    """
    samples = generate_samples(n_rows, seed=seed, iters=gen_iters)
    X = np.array([s.feats for s in samples], dtype=np.float32)
    Y = np.array([s.cfv for s in samples], dtype=np.float32)
    M = np.array([s.mask for s in samples], dtype=np.float32)
    # A row whose targets are entirely zero teaches nothing and quietly drags
    # the reported val_loss toward zero, making the net look better the more
    # degenerate its data is. That is the [19]-class failure — a number that
    # improves for the wrong reason — so it is refused rather than averaged in
    # (wave-3 [M14]).
    dead = int((M.sum(axis=1) == 0).sum())
    if dead:
        raise ValueError(
            f"{dead}/{len(M)} training rows have an all-zero target mask: the "
            "subgames produced no valid matchup. Fix generation rather than "
            "training on rows that cannot carry signal.")

    net, val_loss = train_valuenet(X, Y, M, epochs=epochs, seed=seed)

    rng = np.random.default_rng(seed + 1)
    ex_or, ex_net = [], []
    for _ in range(n_eval):
        board4 = _deal_board(rng, 4)
        r0 = sample_range(rng, board4, keep_frac=eval_keep_frac)
        r1 = sample_range(rng, board4, keep_frac=eval_keep_frac)
        if r0.sum() == 0 or r1.sum() == 0:
            continue
        pot0 = float(rng.uniform(2.0, STACK_BB))
        res = evaluate_spot(net, board4, pot0, r0, r1, iters=eval_iters)
        ex_or.append(res["expl_oracle_bb"])
        ex_net.append(res["expl_net_bb"])
    # Degenerate eval draws are skipped above, so the loop can silently end up
    # averaging over far fewer spots than asked for. A verdict computed from a
    # handful of subgames is not a verdict (wave-3 [M14]).
    if len(ex_net) < n_eval:
        raise ValueError(
            f"only {len(ex_net)}/{n_eval} eval spots produced a valid matchup; "
            "the go/no-go would be averaging over a smaller sample than "
            "requested. Raise n_eval or eval_keep_frac.")

    mean_or = float(np.mean(ex_or)); mean_net = float(np.mean(ex_net))
    return SpikeVerdict(
        n_rows=len(samples), val_loss=val_loss, n_eval=len(ex_net),
        mean_expl_oracle_bb=mean_or, mean_expl_net_bb=mean_net,
        tolerance=tolerance, go=mean_net <= mean_or * tolerance,
    )
