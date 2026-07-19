"""Slice G — the PBS value net learns to predict counterfactual values (§3).

Behavior 3: a small MLP trained on self-generated (PBS → equilibrium CFV) data
predicts held-out counterfactual values. The fast test trains on a checked-in
mini fixture (deterministic, CPU) and asserts the net actually fits; the full
dataset + end-to-end NashConv run lives in the training script / bench.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from pokerlab.rebel.valuenet import (
    FEAT_DIM,
    OUT_DIM,
    PBSValueNet,
    net_leaf_value_fn,
    train_valuenet,
)

FIXTURE = Path(__file__).parent / "fixtures" / "leduc_pbs_mini.npz"


def _load():
    d = np.load(FIXTURE)
    return d["features"], d["targets"]


def test_fixture_has_expected_feature_and_target_dims():
    x, y = _load()
    assert x.shape[1] == FEAT_DIM and y.shape[1] == OUT_DIM
    assert x.shape[0] == y.shape[0] > 100


def test_valuenet_fits_self_generated_cfv_targets():
    x, y = _load()
    net, rep = train_valuenet(x, y, hidden=64, epochs=300, seed=0, device="cpu")
    # A raw (mean-predictor) baseline RMSE ~ target std; the net must beat it hard.
    baseline = float(y.std())
    assert rep.val_rmse < 0.25 * baseline
    assert rep.val_rmse < 0.03
    assert rep.device == "cpu"


def test_training_is_deterministic_under_seed():
    x, y = _load()
    _, r1 = train_valuenet(x, y, epochs=60, seed=3, device="cpu")
    _, r2 = train_valuenet(x, y, epochs=60, seed=3, device="cpu")
    assert r1.val_rmse == r2.val_rmse


def test_leaf_value_fn_returns_per_card_values_shape():
    net = PBSValueNet()
    lf = net_leaf_value_fn(net)
    out = lf((3, 3), (2, 1), np.ones(6), np.ones(6))
    assert out.shape == (2, 6)
    assert np.isfinite(out).all()


@pytest.mark.slow
def test_net_driven_trunk_reproduces_full_game_root_value(leduc_sigma_star):
    # Amended-A exit (M5): a value net trained on self-generated σ*-continuation
    # data drives the depth-limited trunk to reproduce the full-game ROOT VALUE to
    # within 2e-3 — the net is an accurate drop-in for the oracle leaf values.
    from pokerlab.cfr.exploit import on_policy_values
    from pokerlab.cfr.game import build_tree
    from pokerlab.rebel.dataset import generate_dataset
    from pokerlab.rebel.depth_limited import DepthLimitedLeducOracle
    from pokerlab.rebel.trunk import DepthLimitedSolver

    full_tree, sigma = leduc_sigma_star
    v_full = on_policy_values(full_tree, sigma)[0]

    x, y = generate_dataset(n_per_line=4000, seed=0)
    assert x.shape[1] == FEAT_DIM and y.shape[1] == OUT_DIM
    net, rep = train_valuenet(x, y, epochs=300, seed=0, device="cpu")
    assert rep.val_rmse < 0.01

    trunk = DepthLimitedSolver(net_leaf_value_fn(net))
    trunk.run(600)
    dl_tree = build_tree(DepthLimitedLeducOracle(sigma))
    v_net = on_policy_values(dl_tree, trunk.round1_profile())[0]
    assert abs(v_net - v_full) <= 2e-3
