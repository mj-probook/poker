"""Slice C — jam/fold Nash chart engine (impl doc §3 Slice C; plan §5.1).

Three things are proven here:
  1. the numpy solver's math == a real Slice-B game tree (cross-check vs
     cfr.CFRSolver + cfr.exploit on a tiny fixture) — the "test BR first on a
     tiny fixture" requirement;
  2. the full 169-hand chip-EV profile is a Nash equilibrium (exploitability→0);
  3. the solved ranges reproduce well-known published push/fold facts.
"""

import numpy as np
import pytest

from pokerlab.cfr import CFRSolver, build_tree, exploitability, on_policy_values
from pokerlab.charts import hands
from pokerlab.charts.equity import load_equity_matrix
from pokerlab.charts.jamfold import (
    JamFoldGame,
    chip_model,
    jamfold_range,
    joint_prior,
    model_exploitability,
    restrict_model,
    solve_jamfold,
    solve_model,
)

DEPTHS = (5, 8, 10, 15, 20)


def _numpy_profile_as_cfr(x, y, n):
    """Numpy (jam,call) frequencies -> a cfr Profile over JamFoldGame infosets."""
    prof = {}
    for i in range(n):
        prof[f"SB:{i}"] = {"jam": float(x[i]), "fold": float(1 - x[i])}
        prof[f"BB:{i}"] = {"call": float(y[i]), "fold": float(1 - y[i])}
    return prof


def test_numpy_solver_matches_cfr_on_tiny_fixture():
    # Restrict the real chip model to 3 well-separated classes at 10bb and solve
    # both ways. The independent Slice-B instrument must agree.
    E = load_equity_matrix().equity_matrix
    idx = [hands.HAND_INDEX[h] for h in ("AA", "QQ", "72o")]
    model = restrict_model(chip_model(10.0, 0.0, E), idx)
    P = joint_prior()[np.ix_(idx, idx)]
    P = P / P.sum()

    x, y, w = solve_model(model, P, iters=4000)
    expl_np = model_exploitability(model, P, x, y, w)

    game = JamFoldGame(model, P)
    tree = build_tree(game)
    solver = CFRSolver(tree, plus=True)
    solver.run(4000)
    prof = solver.average_profile()
    expl_cfr = exploitability(tree, prof)

    # both are equilibria
    assert expl_np < 1e-4
    assert expl_cfr < 1e-4
    # game value (unique for zero-sum) agrees between the two implementations
    sb_val_cfr = on_policy_values(tree, prof)[0]
    sb_val_np = float((x * (P * (y[None, :] * model.call_sb
                 + (1 - y)[None, :] * model.bbfold_sb)).sum(axis=1)
                 + (1 - x) * (P.sum(axis=1) * model.sbfold_sb)).sum())
    assert sb_val_cfr == pytest.approx(sb_val_np, abs=1e-3)
    # THE key check: numpy's own strategy, judged by the independent instrument,
    # is itself a near-equilibrium.
    expl_of_numpy = exploitability(tree, _numpy_profile_as_cfr(x, y, len(idx)))
    assert expl_of_numpy < 1e-3


@pytest.mark.parametrize("depth", DEPTHS)
def test_full_chip_profile_is_nash(depth):
    # 169-hand chip-EV solve: exploitability ≈ 0 within the model.
    #
    # Bar is 1e-5, not the 1e-3 this asserted through wave 3. PLAN §8 M1.5
    # states the exit as "~1e-6 on the two-action game" and the solve meets it
    # (measured 7.12e-7 at 5bb rising to 2.68e-6 at 20bb), but a 1e-3 assertion
    # would have stayed green through a 370x regression -- passing CI while
    # violating the stated exit by three orders ([R4-3]). The solve is
    # deterministic, so the remaining 3.7x is headroom for float drift across
    # platforms, not for behavior.
    sol = solve_jamfold(float(depth))
    assert sol.exploitability < 1e-5


def test_ranges_shrink_monotonically_with_depth():
    jam = [solve_jamfold(float(d)).sb_jam_combos() for d in DEPTHS]
    call = [solve_jamfold(float(d)).bb_call_combos() for d in DEPTHS]
    assert jam == sorted(jam, reverse=True)   # deeper -> tighter jam
    assert call == sorted(call, reverse=True)  # deeper -> tighter call


def test_published_fact_all_pairs_jam_at_10bb():
    sol = solve_jamfold(10.0)
    for r in "AKQJT98765432":
        assert sol.sb_jam[hands.HAND_INDEX[r + r]] >= 0.5, f"{r+r} should jam @10bb"


def test_published_fact_all_aces_jam_at_10bb():
    sol = solve_jamfold(10.0)
    for r in "KQJT98765432":
        assert sol.sb_jam[hands.HAND_INDEX[f"A{r}s"]] >= 0.5
        assert sol.sb_jam[hands.HAND_INDEX[f"A{r}o"]] >= 0.5


def test_published_fact_72o_folds_from_15bb():
    # 72o is the worst hand — jams only at the very shallowest stacks.
    for depth in (10, 15, 20):
        assert solve_jamfold(float(depth)).sb_jam[hands.HAND_INDEX["72o"]] < 0.5


def test_published_fact_bb_always_calls_premiums():
    for depth in DEPTHS:
        sol = solve_jamfold(float(depth))
        assert sol.bb_call[hands.HAND_INDEX["AA"]] >= 0.99
        assert sol.bb_call[hands.HAND_INDEX["KK"]] >= 0.99


def test_range_widths_in_published_ballpark_at_10bb():
    sol = solve_jamfold(10.0)
    jam_pct = sol.sb_jam_combos() / 1326
    call_pct = sol.bb_call_combos() / 1326
    assert 0.52 <= jam_pct <= 0.68     # HU Nash SB open-jam ~58% at 10bb
    assert 0.32 <= call_pct <= 0.44    # HU Nash BB call ~37% at 10bb


def test_jamfold_range_api_returns_chart_solutions():
    sb = jamfold_range("SB", 10)
    assert len(sb) == 169
    aa = sb["AA"]
    assert aa.source == "chart"
    assert set(aa.actions) == {"jam", "fold"}
    # frequencies form a distribution
    assert sum(f for _, f in aa.actions.values()) == pytest.approx(1.0)
    assert aa.actions["jam"][1] >= 0.99   # AA always jams
    # BB side exposes call/fold with real EVs (calling AA beats folding it)
    bb = jamfold_range("BB", 10)
    assert set(bb["AA"].actions) == {"call", "fold"}
    assert bb["AA"].actions["call"][1] >= 0.99
    assert bb["AA"].actions["call"][0] > bb["AA"].actions["fold"][0]


def test_jamfold_range_rejects_bad_position():
    with pytest.raises(ValueError):
        jamfold_range("BTN", 10)
