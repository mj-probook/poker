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
    assert X.shape == (6, hunl.FEAT_DIM) and Y.shape == (6, hunl.N_CLASSES)
    assert M.shape == (6, hunl.N_CLASSES)


def test_valuenet_trains_and_predicts():
    samples = hunl.generate_samples(24, seed=2, iters=30)
    X = np.array([s.feats for s in samples]); Y = np.array([s.cfv for s in samples])
    M = np.array([s.mask for s in samples])
    net, val_loss = hunl.train_valuenet(X, Y, M, epochs=80, seed=0)
    assert np.isfinite(val_loss) and val_loss >= 0.0
    pred = hunl.net_class_cfv(net, tuple(range(5)), 10.0,
                              np.ones(hunl.NUM_COMBOS), np.ones(hunl.NUM_COMBOS))
    assert pred.shape == (hunl.N_CLASSES,) and np.isfinite(pred).all()


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
