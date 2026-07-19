"""HU postflop subgame solver — vectorized CFR+ (plan §4; impl doc §3 Slice D).

The M3 core: a heads-up (or HU-collapsed) postflop solver over full
1326-combo ranges, solved with **CFR+** in *vector form* — every decision node
carries a ``[num_actions × 1326]`` regret/strategy array and the whole 1326-combo
range is updated per traversal with numpy. Card removal is exact: opponent reach
is combined at every terminal with the card-blocker correction, and chance nodes
zero combos that collide with the dealt runout.

Vertical build order (each street verified before the next):
  1. **river** (fixed 5-card board, no chance) — verified against a brute-force
     best response on tiny ranges (tests/test_solver_subgame_river.py);
  2. **turn+river** (one river chance layer);
  3. **flop** (turn + river chance layers).

The exploitability instrument is a *own* vectorized best response
(`SubgameSolver.best_response` / `.exploitability`), itself verified against the
brute-force BR before it is trusted — the instrument outranks the solver.

Units are big blinds throughout. Payoffs are subgame chip deltas: the pot
(dead money + subgame investment) goes to the winner; a player's payoff is
``winnings - own_subgame_investment``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from pokerlab.charts import hands
from pokerlab.engine.evaluator import rank_showdown
from pokerlab.types import Card

# ---------------------------------------------------------------------------
# Combos: the 1326 unordered hole-card pairs, in a fixed canonical order.
# ---------------------------------------------------------------------------
SOLVER_VERSION = "subgame-cfrplus-v1"


class DegenerateRangeError(ValueError):
    """The subgame has no legal hero/villain matchup, so values are undefined.

    Raised instead of dividing by a zero normalizer. A ValueError subclass so
    existing broad guards still catch it, but named so callers can tell "this
    spot is unsolvable as posed" apart from "the solver hit a bug" — the
    distinction the HH batch drain needs to avoid silent, permanent loss.
    """

COMBOS: tuple[tuple[int, int], ...] = tuple(itertools.combinations(range(52), 2))
NUM_COMBOS = len(COMBOS)  # 1326
COMBO_INDEX: dict[tuple[int, int], int] = {c: i for i, c in enumerate(COMBOS)}
_C1 = np.array([c[0] for c in COMBOS], dtype=np.int64)
_C2 = np.array([c[1] for c in COMBOS], dtype=np.int64)


def uniform_range() -> np.ndarray:
    """Weight 1.0 on every combo (caller masks by board)."""
    return np.ones(NUM_COMBOS, dtype=np.float64)


def range_from_classes(weights: dict[str, float]) -> np.ndarray:
    """Build a 1326 combo-weight vector from a {hand_class: weight} dict.

    Each class' weight is spread uniformly across its concrete combos, matching
    how a range is actually held (a suited class is 4 combos, a pair 6, etc.).
    """
    r = np.zeros(NUM_COMBOS, dtype=np.float64)
    for label, w in weights.items():
        for c in hands.card_combos(label):
            key = (min(c), max(c))
            r[COMBO_INDEX[key]] += w
    return r


def board_mask(board: tuple[Card, ...]) -> np.ndarray:
    """Bool[1326]: True for combos that do not collide with the board."""
    mask = np.ones(NUM_COMBOS, dtype=bool)
    for c in board:
        mask &= (_C1 != c) & (_C2 != c)
    return mask


_STRENGTH_CACHE: dict[tuple[int, ...], np.ndarray] = {}


def hand_strengths(board: tuple[Card, ...]) -> np.ndarray:
    """int32[1326]: showdown rank per combo (smaller = stronger). Blocked
    combos (colliding with the board) get a large sentinel; their reach is 0."""
    key = tuple(sorted(board))
    cached = _STRENGTH_CACHE.get(key)
    if cached is not None:
        return cached
    out = np.full(NUM_COMBOS, 1 << 30, dtype=np.int64)
    mask = board_mask(board)
    for i in range(NUM_COMBOS):
        if mask[i]:
            out[i] = rank_showdown((COMBOS[i][0], COMBOS[i][1]), board)
    _STRENGTH_CACHE[key] = out
    return out


# ---------------------------------------------------------------------------
# Showdown: opponent reach that each combo beats / ties / is valid against,
# with the exact card-removal correction. Vectorized, O(N log N).
# ---------------------------------------------------------------------------
def _card_sums(reach: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    return (np.bincount(c1, weights=reach, minlength=52)
            + np.bincount(c2, weights=reach, minlength=52))


def valid_reach(reach: np.ndarray, c1: np.ndarray = _C1, c2: np.ndarray = _C2) -> np.ndarray:
    """For each combo h: opponent reach mass on combos disjoint from h.

    ``c1``/``c2`` default to the full 1326-combo card arrays; a solver working on
    a board-live subset passes its own subset card arrays.
    """
    cs = _card_sums(reach, c1, c2)
    return reach.sum() - cs[c1] - cs[c2] + reach


class _ShowdownCtx:
    """Per-board, reach-independent precompute for the vectorized showdown.

    The strength sort and group boundaries depend only on the board, so they are
    built once per board and reused across every CFR iteration. Operates on
    whatever combo set (full 1326 or a board-live subset) the caller supplies via
    ``c1``/``c2``.
    """

    def __init__(self, strengths: np.ndarray, c1: np.ndarray = _C1, c2: np.ndarray = _C2):
        n = strengths.size
        self.n = n
        self.c1 = c1
        self.c2 = c2
        order = np.argsort(strengths, kind="stable")
        self.order = order
        s = strengths[order]
        self.a = c1[order]
        self.b = c2[order]
        self.left = np.searchsorted(s, s, side="left")
        self.right = np.searchsorted(s, s, side="right")
        self.idx = np.arange(n)

    def reach(self, reach: np.ndarray):
        cs = _card_sums(reach, self.c1, self.c2)
        total = reach.sum()
        valid = total - cs[self.c1] - cs[self.c2] + reach
        r = reach[self.order]
        a, b, left, right = self.a, self.b, self.left, self.right
        prefix = np.concatenate(([0.0], np.cumsum(r)))
        group_incl = prefix[right] - prefix[left]
        # per-card cumulative reach along strength order (direct assignment: each
        # column touches exactly its two card-rows, so no accumulation needed)
        contrib = np.zeros((52, self.n))
        contrib[a, self.idx] = r
        contrib[b, self.idx] = r
        cardprefix = np.concatenate([np.zeros((52, 1)), np.cumsum(contrib, axis=1)], axis=1)
        sc_a = cardprefix[a, left]; sc_b = cardprefix[b, left]
        gc_a = cardprefix[a, right] - sc_a; gc_b = cardprefix[b, right] - sc_b
        weaker_after = total - prefix[right]
        win_s = weaker_after - (cs[a] - sc_a - gc_a) - (cs[b] - sc_b - gc_b)
        tie_s = group_incl - gc_a - gc_b + r
        win = np.empty(self.n); tie = np.empty(self.n)
        win[self.order] = win_s
        tie[self.order] = tie_s
        return win, tie, valid


def showdown_reach(strengths: np.ndarray, reach: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(win, tie, valid) opponent-reach vectors for each combo, disjoint-correct.

    win[h]  = Σ reach over disjoint opp combos that h beats (opp strictly weaker)
    tie[h]  = Σ reach over disjoint opp combos of equal strength
    valid[h]= Σ reach over all disjoint opp combos  (= win + lose + tie)
    """
    return _ShowdownCtx(strengths).reach(reach)


def showdown_reach_naive(strengths: np.ndarray, reach: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """O(N^2) reference showdown — obviously correct, used to verify the fast
    path in tests. Only call on small live ranges."""
    n = strengths.size
    live = np.nonzero(reach > 0)[0]
    win = np.zeros(n)
    tie = np.zeros(n)
    valid = np.zeros(n)
    for h in range(n):
        a1, a2 = _C1[h], _C2[h]
        for hp in live:
            if _C1[hp] in (a1, a2) or _C2[hp] in (a1, a2):
                continue  # share a card
            valid[h] += reach[hp]
            if strengths[h] < strengths[hp]:
                win[h] += reach[hp]
            elif strengths[h] == strengths[hp]:
                tie[h] += reach[hp]
    return win, tie, valid


# ---------------------------------------------------------------------------
# Betting tree: decision / chance / terminal nodes (HU, OOP=0 acts first).
# ---------------------------------------------------------------------------
@dataclass
class Terminal:
    kind: str  # "fold" | "showdown"
    pot: float
    committed: tuple[float, float]  # subgame-added chips per player
    winner: int | None = None  # fold winner


@dataclass
class Chance:
    children: list[tuple[int, "Node"]]  # (dealt card, subtree)
    divisor: float
    # Public state at this deal, recorded for depth-limited solvers that replace
    # a chance node with a learned leaf and must query it at the node's OWN
    # state. Both players have matched whenever a street closes, so a single
    # `stack` (chips behind, per player) is exact.
    pot: float = 0.0
    stack: float = 0.0


@dataclass
class Decision:
    player: int
    actions: list[str]
    children: list["Node"]
    nid: int = -1


Node = Terminal | Chance | Decision


@dataclass
class BetConfig:
    """Postflop betting shape. Sizes are pot-fractions; ``jam`` adds all-in."""

    sizes: tuple[float, ...] = (0.33, 0.75, 1.25)
    jam: bool = True
    max_raises: int = 2  # aggressive actions beyond the first bet, per street


def _remaining(board: tuple[int, ...]) -> list[int]:
    used = set(board)
    return [c for c in range(52) if c not in used]


class _TreeBuilder:
    def __init__(self, pot0: float, stack: float, cfg: BetConfig):
        self.pot0 = pot0
        self.stack0 = stack
        self.cfg = cfg

    # ---- runout with no betting (post all-in) ----
    def _runout(self, board, invested):
        pot = self.pot0 + invested[0] + invested[1]
        if len(board) >= 5:
            return Terminal("showdown", pot, (invested[0], invested[1]))
        divisor = 52 - len(board) - 4
        children = [
            (c, self._runout(tuple(board) + (c,), invested)) for c in _remaining(board)
        ]
        return Chance(children, float(divisor), pot, 0.0)  # all-in: nothing behind

    # ---- what happens when a street closes peacefully (both matched) ----
    def _advance(self, board, invested, stack):
        pot = self.pot0 + invested[0] + invested[1]
        if len(board) >= 5:
            return Terminal("showdown", pot, (invested[0], invested[1]))
        divisor = 52 - len(board) - 4
        children = [
            (c, self._street_start(tuple(board) + (c,), invested, stack))
            for c in _remaining(board)
        ]
        return Chance(children, float(divisor), pot, float(min(stack)))

    def _street_start(self, board, invested, stack):
        return self._node(board, list(invested), list(stack), [0.0, 0.0], 0, 0, False)

    def build(self, board):
        return self._street_start(tuple(board), [0.0, 0.0], [self.stack0, self.stack0])

    # ---- one decision point within a street ----
    def _node(self, board, invested, stack, street_bet, to_act, num_raises, faced):
        cur = max(street_bet)
        tc = cur - street_bet[to_act]
        opp = 1 - to_act
        pot_now = self.pot0 + invested[0] + invested[1]
        actions: list[str] = []
        children: list[Node] = []

        def close(inv, stk):
            if stk[0] <= 1e-9 or stk[1] <= 1e-9:  # someone all-in -> runout
                return self._runout(board, inv)
            return self._advance(board, inv, stk)

        if tc <= 1e-9:  # no bet outstanding: check or bet
            if to_act == 0:
                child = self._node(board, invested, stack, street_bet, 1, num_raises, False)
            else:
                child = close(invested, stack)  # check-through closes the street
            actions.append("check")
            children.append(child)
            self._add_bets(board, invested, stack, street_bet, to_act, num_raises,
                           pot_now, add=0.0, actions=actions, children=children)
        else:  # facing a bet: fold / call / raise
            actions.append("fold")
            children.append(Terminal("fold", pot_now, (invested[0], invested[1]), winner=opp))
            call_amt = min(tc, stack[to_act])
            inv2 = list(invested); stk2 = list(stack); sb2 = list(street_bet)
            inv2[to_act] += call_amt; stk2[to_act] -= call_amt; sb2[to_act] += call_amt
            actions.append("call")
            children.append(close(inv2, stk2))
            self._add_bets(board, invested, stack, street_bet, to_act, num_raises,
                           pot_now, add=tc, actions=actions, children=children)

        return Decision(to_act, actions, children)

    def _add_bets(self, board, invested, stack, street_bet, to_act, num_raises,
                  pot_now, add, actions, children):
        """Append bet/raise/jam children. ``add`` is the call cost already owed
        (0 for an open bet, ``tc`` for a raise)."""
        if num_raises >= self.cfg.max_raises and add > 0:
            # raise cap reached; only jam allowed beyond the cap
            self._maybe_jam(board, invested, stack, street_bet, to_act, num_raises,
                            actions, children)
            return
        pot_after_call = pot_now + add
        seen: set[float] = set()
        for f in self.cfg.sizes:
            raise_by = f * pot_after_call
            total_add = add + raise_by
            if total_add >= stack[to_act] - 1e-9:
                continue  # would be a jam; handled separately
            amt = round(total_add, 6)
            if amt <= add + 1e-9 or amt in seen:
                continue
            seen.add(amt)
            self._apply_aggr(board, invested, stack, street_bet, to_act, num_raises,
                             total_add, f"{'raise' if add > 0 else 'bet'}_{f}", actions, children)
        self._maybe_jam(board, invested, stack, street_bet, to_act, num_raises,
                        actions, children)

    def _maybe_jam(self, board, invested, stack, street_bet, to_act, num_raises,
                   actions, children):
        if not self.cfg.jam:
            return
        jam_add = stack[to_act]
        if jam_add <= max(street_bet) - street_bet[to_act] + 1e-9:
            return  # jam wouldn't even cover the call -> it's just a call
        self._apply_aggr(board, invested, stack, street_bet, to_act, num_raises,
                         jam_add, "jam", actions, children)

    def _apply_aggr(self, board, invested, stack, street_bet, to_act, num_raises,
                    total_add, label, actions, children):
        inv2 = list(invested); stk2 = list(stack); sb2 = list(street_bet)
        add = min(total_add, stack[to_act])
        inv2[to_act] += add; stk2[to_act] -= add; sb2[to_act] += add
        opp = 1 - to_act
        # opponent now faces the (re)raise
        child = self._node(board, inv2, stk2, sb2, opp, num_raises + 1, True)
        actions.append(label)
        children.append(child)


def build_tree(board, pot0, stack, cfg: BetConfig | None = None) -> Node:
    """Build the postflop subgame tree. ``board`` length selects the depth:
    5→river-only, 4→turn+river, 3→flop subgame. OOP (player 0) acts first."""
    cfg = cfg or BetConfig()
    return _TreeBuilder(float(pot0), float(stack), cfg).build(tuple(board))


def _assign_ids(root: Node) -> list[Decision]:
    decisions: list[Decision] = []

    def walk(node: Node):
        if isinstance(node, Decision):
            node.nid = len(decisions)
            decisions.append(node)
            for c in node.children:
                walk(c)
        elif isinstance(node, Chance):
            for _, c in node.children:
                walk(c)

    walk(root)
    return decisions


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
@dataclass
class SubgameSolver:
    root: Node
    board: tuple[Card, ...]
    range0: np.ndarray  # OOP, full 1326
    range1: np.ndarray  # IP, full 1326
    pot0: float
    decisions: list[Decision] = field(default_factory=list)

    def __post_init__(self):
        self.decisions = _assign_ids(self.root)
        # Public full-1326 ranges (masked by the root board).
        self.range0 = self.range0 * board_mask(self.board)
        self.range1 = self.range1 * board_mask(self.board)
        # Internal work happens only on combos in the union of the two ranges'
        # support. Every other combo has zero reach forever (either it collides
        # with the board or neither player holds it), so it contributes nothing
        # to any value, regret, or card-removal sum — dropping it is exact. This
        # is the difference between a tractable turn/flop solve and an
        # intractable one (and it makes card removal exact over exactly the
        # combos an opponent can actually hold).
        self.live = np.nonzero((self.range0 > 0) | (self.range1 > 0))[0]
        self.c1 = _C1[self.live]
        self.c2 = _C2[self.live]
        self._r0 = self.range0[self.live].copy()
        self._r1 = self.range1[self.live].copy()
        self.regret = [np.zeros((len(d.actions), self.live.size)) for d in self.decisions]
        self.strat_sum = [np.zeros((len(d.actions), self.live.size)) for d in self.decisions]
        self._sd_cache: dict[tuple[int, ...], _ShowdownCtx] = {}
        self._mask_cache: dict[int, np.ndarray] = {}
        self._t = 0

    # ---- per-solver caches over the live subset ----
    def _sd(self, board) -> _ShowdownCtx:
        key = tuple(sorted(board))
        ctx = self._sd_cache.get(key)
        if ctx is None:
            ctx = _ShowdownCtx(hand_strengths(board)[self.live], self.c1, self.c2)
            self._sd_cache[key] = ctx
        return ctx

    def _cmask(self, card: int) -> np.ndarray:
        m = self._mask_cache.get(card)
        if m is None:
            m = (self.c1 != card) & (self.c2 != card)
            self._mask_cache[card] = m
        return m

    # ---- strategy from regret matching+ ----
    @staticmethod
    def _match(reg: np.ndarray) -> np.ndarray:
        pos = np.maximum(reg, 0.0)
        s = pos.sum(axis=0, keepdims=True)
        a = reg.shape[0]
        return np.where(s > 0, pos / np.where(s > 0, s, 1.0), 1.0 / a)

    def _strategy(self, node: Decision) -> np.ndarray:
        return self._match(self.regret[node.nid])

    def _avg_compressed(self) -> list[np.ndarray]:
        out = []
        for s in self.strat_sum:
            tot = s.sum(axis=0, keepdims=True)
            a = s.shape[0]
            out.append(np.where(tot > 0, s / np.where(tot > 0, tot, 1.0), 1.0 / a))
        return out

    def _scatter(self, compressed: list[np.ndarray]) -> list[np.ndarray]:
        """Expand subset [A, L] strategy arrays back to full [A, 1326]."""
        out = []
        for d, comp in zip(self.decisions, compressed):
            a = len(d.actions)
            full = np.full((a, NUM_COMBOS), 1.0 / a)
            full[:, self.live] = comp
            out.append(full)
        return out

    def current_strategies(self) -> list[np.ndarray]:
        return self._scatter([self._strategy(d) for d in self.decisions])

    def average_strategies(self) -> list[np.ndarray]:
        """Average strategy per decision node, as full 1326-wide arrays."""
        return self._scatter(self._avg_compressed())

    # ---- terminal values (v0, v1) counterfactual (opp-reach weighted) ----
    def _terminal_values(self, node: Terminal, board, reach0, reach1):
        pot = node.pot
        c0, c1 = node.committed
        if node.kind == "fold":
            if node.winner == 0:
                v0 = (pot - c0) * valid_reach(reach1, self.c1, self.c2)
                v1 = -c1 * valid_reach(reach0, self.c1, self.c2)
            else:
                v0 = -c0 * valid_reach(reach1, self.c1, self.c2)
                v1 = (pot - c1) * valid_reach(reach0, self.c1, self.c2)
            return v0, v1
        ctx = self._sd(board)
        win0, tie0, val0 = ctx.reach(reach1)  # opp = IP
        win1, tie1, val1 = ctx.reach(reach0)  # opp = OOP
        v0 = -c0 * val0 + pot * (win0 + 0.5 * tie0)
        v1 = -c1 * val1 + pot * (win1 + 0.5 * tie1)
        return v0, v1

    # ---- one CFR+ traversal (both players updated) ----
    def _walk(self, node: Node, board, reach0, reach1, avg=None):
        if isinstance(node, Terminal):
            return self._terminal_values(node, board, reach0, reach1)
        if isinstance(node, Chance):
            v0 = np.zeros(self.live.size); v1 = np.zeros(self.live.size)
            for card, child in node.children:
                m = self._cmask(card)
                cv0, cv1 = self._walk(child, board + (card,), reach0 * m, reach1 * m, avg)
                v0 += cv0 * m
                v1 += cv1 * m
            return v0 / node.divisor, v1 / node.divisor

        p = node.player
        strat = avg[node.nid] if avg is not None else self._strategy(node)
        v0 = np.zeros(self.live.size); v1 = np.zeros(self.live.size)
        child_vp = []
        for a, child in enumerate(node.children):
            if p == 0:
                cv0, cv1 = self._walk(child, board, reach0 * strat[a], reach1, avg)
            else:
                cv0, cv1 = self._walk(child, board, reach0, reach1 * strat[a], avg)
            vp = cv0 if p == 0 else cv1
            child_vp.append(vp)
            if p == 0:
                v0 += strat[a] * cv0
                v1 += cv1
            else:
                v1 += strat[a] * cv1
                v0 += cv0

        if avg is None:  # training: update regret + strategy sum
            node_val = v0 if p == 0 else v1
            reg = self.regret[node.nid]
            for a in range(len(node.children)):
                reg[a] += child_vp[a] - node_val
            np.maximum(reg, 0.0, out=reg)  # CFR+
            reach_p = reach0 if p == 0 else reach1
            self.strat_sum[node.nid] += (self._t + 1) * reach_p * strat  # linear averaging
        return v0, v1

    def iterate(self, n: int) -> None:
        for _ in range(n):
            self._walk(self.root, self.board, self._r0.copy(), self._r1.copy())
            self._t += 1

    # ---- best response (vectorized) for br_player vs a fixed profile ----
    def _br_walk(self, node: Node, board, reach_opp, br, avg):
        if isinstance(node, Terminal):
            return self._terminal_value_br(node, board, reach_opp, br)
        if isinstance(node, Chance):
            v = np.zeros(self.live.size)
            for card, child in node.children:
                m = self._cmask(card)
                v += self._br_walk(child, board + (card,), reach_opp * m, br, avg) * m
            return v / node.divisor
        p = node.player
        if p == br:
            vals = [self._br_walk(child, board, reach_opp, br, avg) for child in node.children]
            return np.maximum.reduce(vals)
        strat = avg[node.nid]
        v = np.zeros(self.live.size)
        for a, child in enumerate(node.children):
            v += self._br_walk(child, board, reach_opp * strat[a], br, avg)
        return v

    def _terminal_value_br(self, node: Terminal, board, reach_opp, br):
        # reach_opp is the *opponent* of br; return br's counterfactual value.
        pot = node.pot
        cbr = node.committed[br]
        if node.kind == "fold":
            sign = (pot - cbr) if node.winner == br else -cbr
            return sign * valid_reach(reach_opp, self.c1, self.c2)
        win, tie, val = self._sd(board).reach(reach_opp)
        return -cbr * val + pot * (win + 0.5 * tie)

    def _root_norm(self) -> float:
        # total valid joint mass (both hole pairs disjoint)
        norm = float((self._r0 * valid_reach(self._r1, self.c1, self.c2)).sum())
        if norm <= 0.0:
            # Not necessarily an empty range: the two ranges can be non-empty
            # and still share every card, so no legal matchup exists (e.g. hero
            # holds the only combo villain's range is built from). Every
            # per-hand value is then 0/0 — undefined, not zero — so callers must
            # see a named condition rather than a bare ZeroDivisionError they
            # can only treat as "something broke" (wave-2 [E25]).
            raise DegenerateRangeError(
                f"subgame has no legal matchup: {self.live.size} live combo(s) "
                "but zero valid joint reach (ranges block each other entirely)")
        return norm

    def best_response_value(self, br: int, avg=None) -> float:
        avg = avg if avg is not None else self._avg_compressed()
        reach_opp = (self._r1 if br == 0 else self._r0).copy()
        vbr = self._br_walk(self.root, self.board, reach_opp, br, avg)
        my_range = self._r0 if br == 0 else self._r1
        return float((my_range * vbr).sum()) / self._root_norm()

    def on_policy_value(self, player: int, avg=None) -> float:
        avg = avg if avg is not None else self._avg_compressed()
        v0, v1 = self._walk(self.root, self.board, self._r0.copy(), self._r1.copy(), avg=avg)
        v = v0 if player == 0 else v1
        my_range = self._r0 if player == 0 else self._r1
        return float((my_range * v).sum()) / self._root_norm()

    def exploitability(self, avg=None) -> float:
        """Average per-player best-response gain, in bb."""
        avg = avg if avg is not None else self._avg_compressed()
        gain = 0.0
        for p in (0, 1):
            gain += self.best_response_value(p, avg) - self.on_policy_value(p, avg)
        return gain / 2.0

    def exploitability_pct(self, avg=None) -> float:
        """Exploitability as a fraction of the starting pot."""
        return self.exploitability(avg) / self.pot0

    # ---- root strategy/EVs, per live combo (adapter input) ----
    def root_action_evs(self, player: int = 0, avg=None):
        """At the root decision node for ``player``: action labels, per-action
        bb-EV per live combo, per-action frequency per live combo, and that
        player's reach. The root actor is the first decision node (OOP=0)."""
        avg = avg if avg is not None else self._avg_compressed()
        root = self.decisions[0]
        if root.player != player:
            raise ValueError(f"root actor is player {root.player}, not {player}")
        opp_reach = self._r1 if player == 0 else self._r0
        valid_opp = valid_reach(opp_reach, self.c1, self.c2)
        safe = np.where(valid_opp > 0, valid_opp, 1.0)
        labels = list(root.actions)
        evs = []
        for child in root.children:
            v0, v1 = self._walk(child, self.board, self._r0.copy(), self._r1.copy(), avg=avg)
            v = v0 if player == 0 else v1
            evs.append(np.where(valid_opp > 0, v / safe, 0.0))
        freqs = avg[root.nid]  # [num_actions, live]
        my_reach = self._r0 if player == 0 else self._r1
        return labels, evs, freqs, my_reach
