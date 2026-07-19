"""Slice I items 1–2: tiny-HUNL depth-limited value-net spike.

Fast tests cover the pieces (PBS features, oracle CFV labels, dataset+Parquet,
net training, the depth-limited eval mechanism). The full L4 go/no-go run is
slow-marked; its recorded verdict lives in docs/notes/tiny-hunl-spike.md.
"""

import numpy as np
import pytest

from pokerlab.spike import hunl


def test_class_bridge_and_features():
    assert (hunl._COMBO_TO_CLASS >= 0).all()
    board = (0, 1, 2, 3, 4)
    r = np.ones(hunl.NUM_COMBOS)
    cls = hunl.class_reach(r * 0 + 1)
    assert cls.shape == (hunl.N_CLASSES,) and cls.sum() == hunl.NUM_COMBOS
    f = hunl.features(board, 10.0, r.copy(), r.copy())
    assert f.shape == (hunl.FEAT_DIM,) and np.isfinite(f).all()


def test_oracle_cfv_labels_have_shape_and_mask():
    rng = np.random.default_rng(0)
    board = hunl._deal_board(rng, 5)
    r0 = hunl.sample_range(rng, board)
    r1 = hunl.sample_range(rng, board)
    s = hunl.solve_river(board, 10.0, r0, r1, iters=40)
    cfv, mask = hunl.oop_class_cfv(s)
    assert cfv.shape == (hunl.N_CLASSES,) and mask.shape == (hunl.N_CLASSES,)
    assert mask.max() == 1.0 and np.isfinite(cfv).all()


def test_dataset_parquet_roundtrip(tmp_path):
    samples = hunl.generate_samples(6, seed=1, iters=25)
    assert len(samples) == 6
    path = hunl.write_parquet(samples, tmp_path / "ds.parquet")
    X, Y, M = hunl.read_parquet(path)
    assert X.shape == (6, hunl.FEAT_DIM) and Y.shape == (6, hunl.OUT_DIM)
    assert M.shape == (6, hunl.OUT_DIM)


def test_valuenet_trains_and_predicts():
    samples = hunl.generate_samples(24, seed=2, iters=30)
    X = np.array([s.feats for s in samples]); Y = np.array([s.cfv for s in samples])
    M = np.array([s.mask for s in samples])
    net, val_loss = hunl.train_valuenet(X, Y, M, epochs=80, seed=0)
    assert np.isfinite(val_loss) and val_loss >= 0.0
    pred = hunl.net_class_cfv(net, tuple(range(5)), 10.0,
                              np.ones(hunl.NUM_COMBOS), np.ones(hunl.NUM_COMBOS))
    assert pred.shape == (hunl.OUT_DIM,) and np.isfinite(pred).all()



# --------------------------------------------------------------------------- #
# Round-1 finding [19]: leaf UNITS regression.
#
# The net is trained on NORMALIZED per-hand EV (per-matchup bb), but
# `SubgameSolver._walk` propagates OPPONENT-REACH-WEIGHTED counterfactual values.
# Feeding the raw net output in as a leaf value made the leaf branch ~1-2 orders
# of magnitude smaller than its sibling terminals, so turn CFR effectively
# ignored the net. These tests pin the units at the leaf.
# --------------------------------------------------------------------------- #
class _StubNet:
    """Constant-prediction stand-in for the value net (units test, no training).

    Emits a plausible NORMALIZED per-hand EV (bb per matchup) for every class:
    ``oop_ev`` on the OOP head and ``ip_ev`` on the IP head (when one exists).
    """

    def __init__(self, oop_ev: float, ip_ev: float):
        self.oop_ev = oop_ev
        self.ip_ev = ip_ev

    def __call__(self, x):
        import torch

        out = torch.full((x.shape[0], hunl.OUT_DIM), float(self.ip_ev))
        out[:, : hunl.N_CLASSES] = float(self.oop_ev)
        return out


def _turn_fixture(seed: int = 11, keep_frac: float = 0.06, pot0: float = 8.0):
    """A tiny turn spot: 4-card board + two sparse ranges + a built DLS."""
    rng = np.random.default_rng(seed)
    board4 = hunl._deal_board(rng, 4)
    r0 = hunl.sample_range(rng, board4, keep_frac=keep_frac)
    r1 = hunl.sample_range(rng, board4, keep_frac=keep_frac)
    tree = hunl.sg.build_tree(board4, pot0=pot0, stack=hunl.STACK_BB,
                              cfg=hunl.TINY_CFG)
    dls = hunl.DepthLimitedTurnSolver(tree, board4, r0.copy(), r1.copy(),
                                      pot0=pot0)
    return board4, r0, r1, pot0, dls


def _first_chance(node):
    """The first river-entry chance node — the one the net leaf replaces."""
    from pokerlab.solver.subgame import Chance, Decision

    if isinstance(node, Chance):
        return node
    if isinstance(node, Decision):
        for c in node.children:
            found = _first_chance(c)
            if found is not None:
                return found
    return None


def test_net_leaf_is_in_counterfactual_units():
    """[19] The net leaf must enter `_walk` on the scale of the branch it stands
    in for.

    A normalized per-hand EV is ~O(pot); the counterfactual values `_walk`
    propagates are that times the opponent's valid-reach mass (tens). Feeding
    the former where the latter is expected silently mutes the net — the leaf
    branch arrives 1-2 orders of magnitude light and CFR ignores it. Reference =
    the exact value of the river chance node the leaf replaces (the branch's own
    siblings straddle 8bb-pot folds and 20bb jams, so the node itself is the
    apples-to-apples scale).
    """
    from pokerlab.solver.subgame import SubgameSolver

    board4, _, _, pot0, dls = _turn_fixture()
    net = _StubNet(oop_ev=0.25 * pot0, ip_ev=0.25 * pot0)

    chance = _first_chance(dls.root)
    assert chance is not None
    ref0, ref1 = SubgameSolver._walk(dls, chance, board4,
                                     dls._r0.copy(), dls._r1.copy(), None)

    leaf = hunl.net_leaf_fn(net, dls, pot0)
    v0, v1 = leaf(board4, dls._r0.copy(), dls._r1.copy())

    for leaf_v, ref in ((v0, ref0), (v1, ref1)):
        ratio = float(np.abs(leaf_v).mean()) / float(np.abs(ref).mean())
        assert 0.2 <= ratio <= 5.0, (
            f"leaf/branch magnitude ratio {ratio:.4g} — leaf values are not "
            "in the opponent-reach-weighted counterfactual units _walk expects"
        )


def test_counterfactual_conversion_inverts_normalization():
    """[19] normalized -> counterfactual is the exact inverse of the labeller's
    normalization, at the leaf's own belief."""
    board4, _, _, pot0, dls = _turn_fixture()
    reach0, reach1 = dls._r0.copy(), dls._r1.copy()
    rng = np.random.default_rng(5)
    v0 = rng.normal(size=dls.live.size) * 10.0
    v1 = rng.normal(size=dls.live.size) * 10.0

    ev0, ev1 = hunl.normalized_from_counterfactual(dls, v0, v1, reach0, reach1)
    b0, b1 = hunl.counterfactual_from_normalized(dls, ev0, ev1, reach0, reach1)

    # combos with no valid opponent hand carry no counterfactual value at all
    valid0 = hunl.sg.valid_reach(reach1, dls.c1, dls.c2)
    valid1 = hunl.sg.valid_reach(reach0, dls.c1, dls.c2)
    np.testing.assert_allclose(b0[valid0 > 0], v0[valid0 > 0], rtol=1e-9)
    np.testing.assert_allclose(b1[valid1 > 0], v1[valid1 > 0], rtol=1e-9)


def test_oracle_leaf_depth_limited_equals_full_solve():
    """[19] Sanity (mirrors Slice G): a depth-limited solve whose leaf returns
    the EXACT continuation — round-tripped through the normalized units the net
    is trained in — must reproduce the full turn+river solve."""
    from pokerlab.solver.subgame import Chance, SubgameSolver

    # the equality is exact at any iteration count, so keep both solves tiny
    board4, r0, r1, pot0, _ = _turn_fixture(seed=13, keep_frac=0.03)

    class _OracleLeafDLS(hunl.DepthLimitedTurnSolver):
        def _walk(self, node, board, reach0, reach1, avg=None):
            if isinstance(node, Chance) and len(board) == 4:
                v0, v1 = SubgameSolver._walk(self, node, board, reach0, reach1, avg)
                ev0, ev1 = hunl.normalized_from_counterfactual(
                    self, v0, v1, reach0, reach1)
                return hunl.counterfactual_from_normalized(
                    self, ev0, ev1, reach0, reach1)
            return SubgameSolver._walk(self, node, board, reach0, reach1, avg)

    iters = 12
    full = hunl.solve_river(board4, pot0, r0, r1, iters=iters)
    tree = hunl.sg.build_tree(board4, pot0=pot0, stack=hunl.STACK_BB,
                              cfg=hunl.TINY_CFG)
    dls = _OracleLeafDLS(tree, board4, r0.copy(), r1.copy(), pot0=pot0)
    dls.iterate(iters)

    assert dls.exploitability() == pytest.approx(full.exploitability(), abs=1e-9)


def test_both_players_have_cfv_targets():
    """[19] IP's leaf value is NOT `-v0`: dead money makes the subgame non-zero-
    sum per matchup, and the two players index different hands. The dataset
    therefore carries a per-player target."""
    rng = np.random.default_rng(4)
    board = hunl._deal_board(rng, 5)
    r0 = hunl.sample_range(rng, board)
    r1 = hunl.sample_range(rng, board)
    s = hunl.solve_river(board, 10.0, r0, r1, iters=40)

    cfv0, mask0 = hunl.class_cfv(s, 0)
    cfv1, mask1 = hunl.class_cfv(s, 1)
    assert cfv0.shape == cfv1.shape == (hunl.N_CLASSES,)
    both = mask0 * mask1 > 0
    assert both.any()
    # not the zero-sum complement of each other
    assert not np.allclose(cfv0[both], -cfv1[both], atol=1e-6)

    # player 0's target is exactly the historical OOP label
    old, old_mask = hunl.oop_class_cfv(s)
    np.testing.assert_allclose(cfv0, old, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(mask0, old_mask)


def test_samples_carry_both_heads():
    """[19] generate_samples emits the stacked two-head target."""
    samples = hunl.generate_samples(3, seed=1, iters=25)
    assert hunl.OUT_DIM == 2 * hunl.N_CLASSES
    for s in samples:
        assert s.cfv.shape == (hunl.OUT_DIM,)
        assert s.mask.shape == (hunl.OUT_DIM,)


@pytest.mark.slow
def test_evaluate_spot_measures_exploitability():
    samples = hunl.generate_samples(120, seed=3, iters=60)
    X = np.array([s.feats for s in samples]); Y = np.array([s.cfv for s in samples])
    M = np.array([s.mask for s in samples])
    net, _ = hunl.train_valuenet(X, Y, M, epochs=200, seed=0)

    rng = np.random.default_rng(9)
    board4 = hunl._deal_board(rng, 4)
    r0 = hunl.sample_range(rng, board4, keep_frac=0.1)
    r1 = hunl.sample_range(rng, board4, keep_frac=0.1)
    res = hunl.evaluate_spot(net, board4, 12.0, r0, r1, iters=150)
    # the exact oracle turn+river is the low-exploitability baseline; the
    # net-driven turn strategy is measured in the FULL game (net turn + exact
    # river continuation) and is at least as exploitable.
    assert res["expl_net_bb"] >= res["expl_oracle_bb"] - 1e-9
    assert np.isfinite(res["expl_net_bb"]) and res["expl_oracle_bb"] < 1.0


@pytest.mark.slow
def test_run_spike_tiny_produces_verdict():
    v = hunl.run_spike(n_rows=80, n_eval=2, seed=0, gen_iters=50,
                       eval_iters=80, epochs=100)
    assert v.n_rows == 80 and v.n_eval == 2
    assert np.isfinite(v.val_loss)
    assert np.isfinite(v.mean_expl_net_bb) and np.isfinite(v.mean_expl_oracle_bb)
    assert isinstance(v.go, bool)
