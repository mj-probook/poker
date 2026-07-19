"""Slice C — equity-matrix GENERATOR internals (impl doc §3 Slice C).

The checked-in npz is validated for accuracy in test_charts_equity; here we test
the sampler that built it directly, so a silent regeneration bug (bad collision
handling / board overlap) can't quietly bias a future rebuild. The statistical
regeneration check is `slow` (it re-runs Monte-Carlo).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pokerlab.charts import hands

_GEN = Path(__file__).resolve().parents[1] / "scripts" / "gen_equity_matrix.py"
_spec = importlib.util.spec_from_file_location("gen_equity_matrix", _GEN)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def test_sample_hole_is_class_consistent():
    rng = np.random.default_rng(0)
    for label in ("AA", "AKs", "AKo", "72o", "T9s"):
        cards = gen._sample_hole(rng, label, 2000)
        hi, lo = hands.hand_ranks(label)
        r = cards // 4
        s = cards % 4
        assert set(np.unique(r + 2)) <= {hi, lo}
        assert (cards[:, 0] != cards[:, 1]).all()
        if hands.is_pair(label):
            assert (r[:, 0] == r[:, 1]).all() and (s[:, 0] != s[:, 1]).all()
        elif label.endswith("s"):
            assert (s[:, 0] == s[:, 1]).all()
        else:
            assert (s[:, 0] != s[:, 1]).all()


def test_boards_never_overlap_holes_and_are_distinct():
    rng = np.random.default_rng(1)
    hi = gen._sample_hole(rng, "AKs", 3000)
    hj = gen._resolve_collisions(rng, "AKs", hi, gen._sample_hole(rng, "AKs", 3000))
    holes = np.concatenate([hi, hj], axis=1)
    # collision resolver produced disjoint hole hands (no shared card, any row)
    shared = (
        (hj[:, 0:1] == hi).any(axis=1) | (hj[:, 1:2] == hi).any(axis=1)
    )
    assert not shared.any()
    board = gen._deal_boards(rng, holes, 3000)
    # 5 distinct board cards, none shared with the 4 holes
    for t in range(0, 3000, 250):
        row = board[t]
        assert len(set(row.tolist())) == 5
        assert set(row.tolist()).isdisjoint(holes[t].tolist())


@pytest.mark.slow
def test_regeneration_matches_checked_in_within_mc_noise():
    # Re-run Monte-Carlo on a few cells with a fresh seed; the checked-in matrix
    # must agree within Monte-Carlo tolerance (artifact not stale/corrupt).
    from pokerlab.charts.equity import load_equity_matrix
    em = load_equity_matrix()
    rng = np.random.default_rng(999)
    for a, b in (("AA", "KK"), ("AKs", "QQ"), ("JTs", "88"), ("A5s", "KQo")):
        w, t = gen._cell_equity(rng, a, b, 20000)
        assert (w + 0.5 * t) == pytest.approx(em.equity(a, b), abs=0.01)
