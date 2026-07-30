"""First-in OPEN game charts (charts/openraise.py): fold/raise/jam, priced.

The jam/fold charts grade raises as off-tree because that game never priced
them. This module solves the bigger game where raises are REAL: the opener
chooses fold / raise-to-r / jam; each seat behind re-jams or folds (vs a
raise) or calls or folds (vs a jam); the opener calls or folds a re-jam.
Every line ends preflop, so the whole game is exactly solvable and carries
the same measured-Nash-gap certificate as the other charts.

Disclosed restrictions (range_ctx): single aggressive responder (no
overcalls), no flat-calls behind (a flat creates a postflop subgame this
engine does not price — same reason limp stays a distractor), pairwise card
removal, symmetric stacks.

Anchor: with NO raise sizes the game IS the jam/fold chain — the solve must
agree with `solve_ring` and certify under ring's own instrument.
"""

import numpy as np
import pytest

from pokerlab.charts import hands
from pokerlab.charts.openraise import RAISE_SIZES, solve_open
from pokerlab.charts.ring import _Chain, _nash_gap, solve_ring

T4 = ("CO", "BTN", "SB", "BB")   # 4-seat table keeps fast tests fast


def test_no_raise_sizes_reduces_to_the_jamfold_chain():
    open_sol = solve_open("CO", 10.0, table=T4, sizes=())
    ring = solve_ring("CO", 10.0, table=T4)
    pure = (ring.jam < 0.01) | (ring.jam > 0.99)
    assert pure.sum() > 100
    assert np.allclose(open_sol.open_freq["jam"][pure], ring.jam[pure],
                       atol=5e-3)
    # cross-certificate: the reduced profile judged by ring's own instrument
    from pokerlab.charts.equity import load_equity_matrix
    from pokerlab.charts.jamfold import joint_prior
    chain = _Chain(T4, "CO", 10.0, 0.0,
                   load_equity_matrix().equity_matrix, joint_prior())
    ys = {q: open_sol.defend_freq[(q, "jam")] for q in ("BTN", "SB", "BB")}
    assert _nash_gap(chain, open_sol.open_freq["jam"], ys) < 1e-4


@pytest.mark.slow
def test_full_open_game_is_certified():
    sol = solve_open("CO", 20.0, table=T4)
    assert sol.exploitability < 5e-5
    # every opener action is priced with a frequency and an EV
    labels = {"jam", "fold"} | {f"raise {r:g}bb" for r in RAISE_SIZES}
    assert set(sol.open_freq) == labels
    assert set(sol.open_ev) == labels
    # frequencies form a distribution per hand
    total = sum(sol.open_freq[a] for a in labels)
    assert np.allclose(total, 1.0, atol=1e-6)


@pytest.mark.slow
def test_aa_never_folds_and_always_calls_a_rejam():
    sol = solve_open("CO", 20.0, table=T4)
    aa = hands.HAND_CLASSES.index("AA")
    assert sol.open_freq["fold"][aa] < 0.01
    for (r, q), z in sol.callback_freq.items():
        assert z[aa] > 0.99


@pytest.mark.slow
def test_resteal_widens_versus_later_position_raises():
    # BB re-jams wider vs a BTN 2.2bb raise than vs a CO 2.2bb raise -- wait,
    # CO is earlier than BTN, so vs BTN must be wider.
    label = f"raise {RAISE_SIZES[0]:g}bb"
    vs_co = solve_open("CO", 20.0, table=T4)
    vs_btn = solve_open("BTN", 20.0, table=("BTN", "SB", "BB"))
    def combos(freqs):
        return float(sum(f * hands.combos(h)
                         for f, h in zip(freqs, hands.HAND_CLASSES)))
    assert (combos(vs_btn.defend_freq[("BB", label)])
            > combos(vs_co.defend_freq[("BB", label)]))


@pytest.mark.slow
def test_deep_hu_sb_actually_raises_sometimes():
    # at 20bb HU the open game is not pure jam/fold — published theory has
    # raise-first ranges at this depth; a solve where raising never happens
    # would mean the raise lines are mispriced
    sol = solve_open("SB", 20.0, table=("SB", "BB"))
    raise_mass = sum(sol.open_freq[f"raise {r:g}bb"].sum()
                     for r in RAISE_SIZES)
    assert raise_mass > 1.0        # summed over 169 classes
