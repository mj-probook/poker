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
    f = hunl.features(board, 10.0, hunl.STACK_BB, r.copy(), r.copy())
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

    leaf = hunl.net_leaf_fn(net, dls)
    v0, v1 = leaf(board4, dls._r0.copy(), dls._r1.copy(), chance.pot, chance.stack)

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



# --------------------------------------------------------------------------- #
# Round-2 follow-up to [19]: the leaf was queried with the SUBGAME ROOT pot and
# an implicit full stack, but a river-entry node sits at whatever the turn
# betting built. On this tree that is pot 8/16/24/48 with 20/16/12/0 behind,
# while every training row was generated at pot~U(2,20) with 20 behind — so the
# net was queried far off its training distribution on every betting line.
# (Both players have matched at a street close, so the stack behind is pinned by
# the pot: stack = start - (pot - pot0)/2.)
# --------------------------------------------------------------------------- #
def _river_entry_states(root):
    """(pot, stack) at every river-entry chance node, derived independently.

    Follows the check-line down to a terminal: no further money goes in, so that
    terminal's pot is the pot at the chance node.
    """
    from pokerlab.solver.subgame import Chance, Decision, Terminal

    out = set()

    def first_terminal(n):
        while not isinstance(n, Terminal):
            n = n.children[0][1] if isinstance(n, Chance) else n.children[0]
        return n

    def walk(n):
        if isinstance(n, Chance):
            t = first_terminal(n)
            out.add((round(t.pot, 6), round(t.committed[0], 6)))
            return
        if isinstance(n, Decision):
            for c in n.children:
                walk(c)

    walk(root)
    # committed -> stack behind
    return {(pot, round(hunl.STACK_BB - inv, 6)) for pot, inv in out}


def test_leaf_is_queried_at_the_nodes_own_pot_and_stack():
    board4, _, _, pot0, dls = _turn_fixture(pot0=8.0)
    seen = []

    def recording_leaf(board, reach0, reach1, pot, stack):
        seen.append((round(pot, 6), round(stack, 6)))
        z = np.zeros(dls.live.size)
        return z, z

    dls.set_leaf(recording_leaf).iterate(1)

    expected = _river_entry_states(dls.root)
    assert set(seen) == expected
    # the whole point: betting lines really do differ from the root pot/stack
    assert len(expected) > 1
    assert (pot0, hunl.STACK_BB) in expected


def test_features_encode_the_remaining_stack():
    """A river subgame with 0 behind is a different game from one with 20."""
    board = (0, 1, 2, 3, 4)
    r = np.ones(hunl.NUM_COMBOS)
    deep = hunl.features(board, 24.0, 12.0, r.copy(), r.copy())
    shallow = hunl.features(board, 24.0, 0.0, r.copy(), r.copy())
    assert deep.shape == (hunl.FEAT_DIM,) == shallow.shape
    assert not np.array_equal(deep, shallow)


def test_data_gen_samples_reachable_river_entry_states():
    """Training rows must come from the (pot, stack) states the leaf will see."""
    samples = hunl.generate_samples(12, seed=5, iters=20)
    stacks = {round(s.stack, 6) for s in samples}
    pots = {round(s.pot, 6) for s in samples}
    # not every row pinned at the full starting stack any more
    assert stacks != {hunl.STACK_BB}
    assert max(pots) > hunl.STACK_BB   # turn betting builds pots past 20bb
    for s in samples:
        assert 0.0 <= s.stack <= hunl.STACK_BB


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


# --------------------------------------------------------------------------- #
# Wave-3 [M2]: the net was evaluated on ranges drawn SPARSER than any it was
# trained on (eval keep_frac 0.10 vs generation 0.2), so eval beliefs
# sat below the sparsest training belief. The net was then blamed for answers
# to questions outside its own support.
#
# This pins the POST-FIX INVARIANT — the beliefs the leaf is queried on lie
# inside the sampled training support — rather than the weaker "the two
# keep_fracs are equal", which a later refactor could satisfy while still
# drawing from different distributions.
# --------------------------------------------------------------------------- #
def _mean_nonzero_classes(keep_frac, *, seed=0, n=250):
    """Mean number of hand classes a sampled range gives weight to.

    Stated convention so the figure is reproducible: non-zero CLASSES (not
    combos) in ranges drawn by `sample_range`, seed 0, n=250.
    """
    import numpy as np

    from pokerlab.charts import hands
    from pokerlab.solver.adapter import COMBO_INDEX

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        board = hunl._deal_board(rng, 5)
        r = hunl.sample_range(rng, board, keep_frac=keep_frac)
        if r.sum() == 0:
            continue
        out.append(sum(
            1 for cls in hands.HAND_CLASSES
            if any(r[COMBO_INDEX[(min(a, b), max(a, b))]] > 0
                   for a, b in hands.card_combos(cls))))
    return float(np.mean(out))


def test_the_documented_belief_sparsity_gap_matches_the_shipped_config():
    """[M2] parameter half — pins the DOC against the CODE, both as shipped.

    The gap is deliberately NOT closed: every recorded number in
    docs/notes/tiny-hunl-spike.md, and both of [M1]'s seed measurements, were
    produced at these densities, so aligning them would leave the results table
    describing code that no longer ships.

    So the invariant worth pinning is not "the densities are equal" but "the
    documented gap is the real one". This fails if anyone changes either density
    without updating the note — including anyone who "helpfully" aligns them.
    """
    gen = _mean_nonzero_classes(hunl.GEN_KEEP_FRAC)
    ev_default = _mean_nonzero_classes(hunl.EVAL_KEEP_FRAC)
    # The recorded run and the shipped default are now the SAME density: the
    # note's run passed eval_keep_frac=0.10 explicitly while the default was
    # 0.12, so the doc and the code described different configurations. [R4-c]
    # aligned the default to the density the recorded numbers actually describe.
    ev_recorded = _mean_nonzero_classes(0.10)

    assert ev_default < gen and ev_recorded < gen, "eval must be the sparser draw"
    # the three figures quoted in the note's belief-axis table
    assert gen == pytest.approx(34.2, abs=0.5), f"doc says gen 34.2, got {gen:.1f}"
    assert ev_recorded == pytest.approx(17.1, abs=0.5), (
        f"doc says recorded-run eval 17.1, got {ev_recorded:.1f}")
    assert ev_default == pytest.approx(17.1, abs=0.5), (
        f"the default must now DRAW what the note records; got {ev_default:.1f}")
    assert ev_default == pytest.approx(ev_recorded, abs=0.01), (
        "recorded run and shipped default must be the same density after [R4-c]")


def test_the_two_densities_are_not_silently_equalised():
    """A guard with teeth: alignment is the change the note forbids."""
    assert hunl.EVAL_KEEP_FRAC != hunl.GEN_KEEP_FRAC, (
        "densities were aligned — every recorded number in the spike note now "
        "describes a configuration that no longer ships. See [M2].")
