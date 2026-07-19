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
def test_self_generation_and_training_pipeline():
    # Exercise the full self-generated pipeline (σ* solve → sample beliefs →
    # label → train), independent of the checked-in fixture.
    from pokerlab.rebel.dataset import generate_dataset

    x, y = generate_dataset(n_per_state=60, seed=1)
    assert x.shape[1] == FEAT_DIM and y.shape[1] == OUT_DIM and x.shape[0] > 120
    _, rep = train_valuenet(x, y, epochs=200, seed=0, device="cpu")
    assert rep.val_rmse < 0.03
