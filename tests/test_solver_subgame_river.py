"""Slice D — river subgame solver + the exploitability instrument (impl §3 D).

The river solver is the base case (fixed 5-card board, no chance). Two things
are pinned here before anything is trusted downstream:

  * the **vectorized showdown** (card-removal correct) matches an O(N^2) naive
    reference exactly, at full 1326-combo scale;
  * the solver's **vectorized best response** matches a brute-force best
    response (independent pure-strategy enumeration) exactly on tiny ranges —
    the instrument is verified before it judges any solver.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from pokerlab.engine.cards import card_from_str
from pokerlab.solver import subgame as sg


def _cards(s: str) -> tuple[int, ...]:
    return tuple(card_from_str(s[i:i + 2]) for i in range(0, len(s), 2))


def _range(combos: list[str]) -> np.ndarray:
    r = np.zeros(sg.NUM_COMBOS)
    for cs in combos:
        a, b = _cards(cs)
        r[sg.COMBO_INDEX[(min(a, b), max(a, b))]] = 1.0
    return r


# ---------------------------------------------------------------------------
# showdown primitive
# ---------------------------------------------------------------------------
def test_combo_enumeration_is_1326():
    assert sg.NUM_COMBOS == 1326


def test_fast_showdown_matches_naive_at_full_scale():
    rng = np.random.default_rng(0)
    for _ in range(4):
        board = tuple(rng.choice(52, size=5, replace=False).tolist())
        strengths = sg.hand_strengths(board)
        reach = rng.random(sg.NUM_COMBOS) * sg.board_mask(board)
        win, tie, valid = sg.showdown_reach(strengths, reach)
        nwin, ntie, nvalid = sg.showdown_reach_naive(strengths, reach)
        assert np.allclose(win, nwin, atol=1e-9)
        assert np.allclose(tie, ntie, atol=1e-9)
        assert np.allclose(valid, nvalid, atol=1e-9)


def test_showdown_win_lose_tie_partition_valid():
    rng = np.random.default_rng(1)
    board = tuple(rng.choice(52, size=5, replace=False).tolist())
    strengths = sg.hand_strengths(board)
    reach = rng.random(sg.NUM_COMBOS) * sg.board_mask(board)
    win, tie, valid = sg.showdown_reach(strengths, reach)
    # lose = valid - win - tie must be >= 0 everywhere
    lose = valid - win - tie
    assert (lose > -1e-9).all()


# ---------------------------------------------------------------------------
# river tree + solver
# ---------------------------------------------------------------------------
BOARD = _cards("AsKd7h2c9s")
R0 = _range(["QhQs", "JdJc", "TdTh"])
R1 = _range(["8d8h", "6c6d", "AcAh"])
CFG = sg.BetConfig(sizes=(0.75,), jam=False, max_raises=0)


def _solver(iters: int = 40):
    tree = sg.build_tree(BOARD, pot0=10.0, stack=20.0, cfg=CFG)
    s = sg.SubgameSolver(tree, BOARD, R0.copy(), R1.copy(), pot0=10.0)
    s.iterate(iters)
    return s


def test_river_tree_is_finite_and_has_decisions():
    tree = sg.build_tree(BOARD, pot0=10.0, stack=20.0, cfg=CFG)
    s = sg.SubgameSolver(tree, BOARD, R0.copy(), R1.copy(), pot0=10.0)
    assert len(s.decisions) > 0


# -- independent brute-force best response by pure-strategy enumeration --
def _support(r):
    return [i for i in range(sg.NUM_COMBOS) if r[i] > 0]


def _br_nodes(root, br):
    out = []

    def walk(n):
        if isinstance(n, sg.Decision):
            if n.player == br:
                out.append(n)
            for c in n.children:
                walk(c)
        elif isinstance(n, sg.Chance):
            for _, c in n.children:
                walk(c)

    walk(root)
    return out


def _terminal_payoff(node, board, h0, h1, br):
    pot = node.pot
    cbr = node.committed[br]
    if node.kind == "fold":
        return (pot - cbr) if node.winner == br else -cbr
    s = sg.hand_strengths(board)
    sbr, sopp = (s[h0], s[h1]) if br == 0 else (s[h1], s[h0])
    if sbr < sopp:
        return pot - cbr
    if sbr == sopp:
        return pot / 2 - cbr
    return -cbr


def _value_br(node, board, h0, h1, br, pure, avg):
    if isinstance(node, sg.Terminal):
        return _terminal_payoff(node, board, h0, h1, br)
    # river tree has no chance nodes
    q = node.player
    if q == br:
        hbr = h1 if br == 1 else h0
        a = pure[(node.nid, hbr)]
        return _value_br(node.children[a], board, h0, h1, br, pure, avg)
    hopp = h0 if br == 1 else h1
    dist = avg[node.nid][:, hopp]
    return sum(
        p * _value_br(child, board, h0, h1, br, pure, avg)
        for p, child in zip(dist, node.children) if p > 0
    )


def _brute_force_br(s: sg.SubgameSolver, br: int) -> float:
    avg = s.average_strategies()
    supp0, supp1 = _support(s.range0), _support(s.range1)
    br_nodes = _br_nodes(s.root, br)
    supp_br = supp1 if br == 1 else supp0
    # every (br node, br combo) infoset chooses an action
    infosets = [(n.nid, h, len(n.children)) for n in br_nodes for h in supp_br]
    norm = float((s.range0 * sg.valid_reach(s.range1)).sum())
    best = -1e18
    for choice in itertools.product(*[range(k) for (_, _, k) in infosets]):
        pure = {(nid, h): a for (nid, h, _), a in zip(infosets, choice)}
        ev = 0.0
        for h0 in supp0:
            for h1 in supp1:
                if len({sg.COMBOS[h0][0], sg.COMBOS[h0][1],
                        sg.COMBOS[h1][0], sg.COMBOS[h1][1]}) < 4:
                    continue  # share a card
                w = s.range0[h0] * s.range1[h1]
                ev += w * _value_br(s.root, s.board, h0, h1, br, pure, avg)
        best = max(best, ev / norm)
    return best


@pytest.mark.parametrize("br", [0, 1])
def test_vectorized_br_matches_brute_force(br):
    s = _solver(iters=25)
    fast = s.best_response_value(br)
    brute = _brute_force_br(s, br)
    assert fast == pytest.approx(brute, abs=1e-9)


def test_river_solver_reduces_exploitability():
    tree = sg.build_tree(BOARD, pot0=10.0, stack=20.0, cfg=CFG)
    s = sg.SubgameSolver(tree, BOARD, R0.copy(), R1.copy(), pot0=10.0)
    s.iterate(1)
    early = s.exploitability_pct()
    s.iterate(1000)
    late = s.exploitability_pct()
    assert late < early
    assert late < 0.005  # ≤0.5% of pot on this tiny river spot


# --------------------------------------------------------------------------- #
# Wave-2 [E25]: a subgame can have live combos but ZERO valid joint mass — every
# hero hand blocks every villain hand. _root_norm() then returned 0.0 and every
# value query died with a bare ZeroDivisionError, which callers could only
# catch as "something went wrong" (and the HH drain turned into permanent,
# invisible loss). It must be a typed, recognisable condition.
# --------------------------------------------------------------------------- #
def _mutually_blocking_solver():
    import numpy as np

    from pokerlab.solver import subgame as sg

    r0 = np.zeros(sg.NUM_COMBOS)
    r1 = np.zeros(sg.NUM_COMBOS)
    r0[sg.COMBO_INDEX[(48, 51)]] = 1.0
    r1[sg.COMBO_INDEX[(48, 50)]] = 1.0   # shares card 48 with the only hero hand
    board = (0, 1, 2, 3, 4)
    tree = sg.build_tree(board, pot0=5.0, stack=10.0)
    return sg.SubgameSolver(tree, board, r0, r1, pot0=5.0)


def test_degenerate_range_is_a_typed_error_not_zero_division():
    from pokerlab.solver.subgame import DegenerateRangeError

    s = _mutually_blocking_solver()
    # the pathology: live combos exist, but no matchup between them is legal
    assert s.live.size >= 1
    s.iterate(2)

    for call in (lambda: s.exploitability(),
                 lambda: s.best_response_value(0),
                 lambda: s.on_policy_value(0),
                 lambda: s.exploitability_pct()):
        with pytest.raises(DegenerateRangeError):
            call()


def test_degenerate_range_error_is_catchable_as_valueerror():
    """Callers that guard broadly should still catch it."""
    s = _mutually_blocking_solver()
    s.iterate(2)
    with pytest.raises(ValueError):
        s.exploitability()
