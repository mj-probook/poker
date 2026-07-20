"""Slice C — preflop equity matrix spot checks (impl doc §3 Slice C).

Tests LOAD the checked-in npz artifact; they never regenerate it (generation is
a script — see scripts/gen_equity_matrix.py). Spot-check tolerances account for
the Monte-Carlo noise of the checked-in trial count.
"""

import numpy as np
import pytest

from pokerlab.charts import hands
from pokerlab.charts.equity import load_equity_matrix


@pytest.fixture(scope="module")
def em():
    return load_equity_matrix()


def test_aa_vs_kk_is_about_0819(em):
    # Textbook pair-over-pair all-in equity.
    assert em.equity("AA", "KK") == pytest.approx(0.819, abs=0.01)


def test_aks_vs_qq_is_about_046(em):
    # Classic "race": suited two overcards slightly behind an under-pair.
    assert em.equity("AKs", "QQ") == pytest.approx(0.46, abs=0.02)


def test_dominated_ace_and_suited_connector_sanity(em):
    # AKo crushes A2o (kicker domination): well above 0.6.
    assert em.equity("AKo", "A2o") > 0.6
    # 22 is a small favourite / coinflip vs two overcards AKo (~0.52).
    assert em.equity("22", "AKo") == pytest.approx(0.52, abs=0.03)


def test_showdown_consistency_win_plus_win_plus_tie_is_one(em):
    # P(a beats b) + P(b beats a) + P(tie) = 1 for every ordered off-diagonal pair.
    n = len(hands.HAND_CLASSES)
    total = em.win + em.win.T + em.tie
    off = ~np.eye(n, dtype=bool)
    assert np.allclose(total[off], 1.0, atol=1e-6)


def test_mirror_symmetry_equity_pair_sums_to_one(em):
    # equity(a,b) + equity(b,a) = 1 exactly (off-diagonal, by construction).
    E = em.equity_matrix
    n = E.shape[0]
    off = ~np.eye(n, dtype=bool)
    assert np.allclose((E + E.T)[off], 1.0, atol=1e-6)


def test_probabilities_in_range_and_tie_symmetric(em):
    assert (em.win >= 0).all() and (em.win <= 1).all()
    assert (em.tie >= 0).all() and (em.tie <= 1).all()
    assert (em.win + em.tie <= 1.0 + 1e-6).all()
    assert np.allclose(em.tie, em.tie.T, atol=1e-6)  # ties are symmetric


def test_diagonal_is_coinflip(em):
    # A class against itself is a coinflip: equity ≈ 0.5 within MC noise.
    for label in ("AA", "72o", "T9s"):
        assert em.equity(label, label) == pytest.approx(0.5, abs=0.03)


def test_labels_align_with_hand_classes(em):
    assert em.labels == hands.HAND_CLASSES
