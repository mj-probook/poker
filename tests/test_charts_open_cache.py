"""The checked-in open-game artifact (data/open_charts.npz).

Same honesty obligation as the ring artifact: the fast suite RE-CERTIFIES
every stored formation from scratch — Nash gap recomputed from the cached
strategies against a freshly built payoff model (the opener's best response
re-optimises its call-vs-re-jam infosets first). Depth 5 stores the EMPTY
raise menu (sizes_for_depth): at 5bb the raise options are near-duplicates
of jam and the honest game is jam/fold.
"""

import numpy as np
import pytest

from pokerlab.charts import hands
from pokerlab.charts.openraise import (OPEN_FORMATIONS, RAISE_SIZES,
                                       _OpenGame, _nash_gap,
                                       load_open_charts, open_solution,
                                       sizes_for_depth, solve_open)
from pokerlab.drills.categories import ANTES, DEPTHS


def test_artifact_covers_the_grid_with_depth_gated_sizes():
    charts = load_open_charts()
    assert len(charts) == len(OPEN_FORMATIONS) * len(DEPTHS) * len(ANTES)
    for formation in OPEN_FORMATIONS:
        assert open_solution(formation, 5.0).sizes == ()
        assert open_solution(formation, 20.0).sizes == RAISE_SIZES
    assert sizes_for_depth(5.0) == ()


def test_every_cached_formation_recertifies_from_scratch():
    from pokerlab.charts.equity import load_equity_matrix
    from pokerlab.charts.jamfold import joint_prior

    E = load_equity_matrix().equity_matrix
    P = joint_prior()
    worst = 0.0
    for sol in load_open_charts().values():
        game = _OpenGame(sol.table, sol.opener, sol.depth_bb, sol.ante,
                         sol.sizes, E, P)
        labels = game.threats + ["fold"]
        x = {lab: sol.open_freq[lab] for lab in labels}
        ys = dict(sol.defend_freq)
        zs = dict(sol.callback_freq)
        worst = max(worst, _nash_gap(game, x, ys, zs))
    # solver stopping rule 1e-5; measured shipped worst 1.00e-05. Margin is
    # for float32 storage + platform drift, not behavior ([R4-3]).
    assert worst < 3e-5


def test_shipped_behavioral_anchors():
    """The published-fact anchors, read from the SHIPPED artifact (cheap)
    — live-solve versions run behind `slow` in test_charts_openraise."""
    aa = hands.HAND_CLASSES.index("AA")
    co20 = open_solution("CO", 20.0)
    assert co20.open_freq["fold"][aa] < 0.01           # AA never folds
    for (r, q), z in co20.callback_freq.items():
        assert z[aa] > 0.99                            # AA calls re-jams
    # deep HU SB genuinely raises
    hu20 = open_solution("SBhu", 20.0)
    raise_mass = sum(hu20.open_freq[f"raise {r:g}bb"].sum()
                     for r in RAISE_SIZES)
    assert raise_mass > 1.0
    # 20bb UTG 9-max: open-jamming is essentially extinct — the raise game
    # replaces it (the measured fact that motivated the whole slice)
    utg20 = open_solution("UTG", 20.0)
    assert utg20.open_freq["jam"].sum() < 0.5