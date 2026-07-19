"""Slice G — Leduc public-belief-state (PBS) encoding (impl doc §3 Slice G).

A PBS pairs a public observation (public card + betting line + contributions)
with each player's *reach* vector over their 6 possible private cards — the
range. The joint belief is the deal prior times both reaches, renormalized with
card-removal (blocking). Fixtures pin shape, normalization, the uniform initial
belief, card blocking, and one hand-computed Bayes step.

Leduc deck: card 0..5, rank = card // 2 (0=J, 1=Q, 2=K), two suits each.
So J = {0, 1}, Q = {2, 3}, K = {4, 5}.
"""

import numpy as np

from pokerlab.rebel.pbs import NUM_CARDS, PBS


def test_initial_reach_is_all_ones_shape_2x6():
    pbs = PBS.initial()
    assert pbs.reach.shape == (2, NUM_CARDS)
    assert np.array_equal(pbs.reach, np.ones((2, NUM_CARDS)))
    assert pbs.public_card is None
    assert pbs.contrib == (1, 1)


def test_initial_joint_belief_is_uniform_over_distinct_pairs():
    # 6*5 = 30 ordered distinct (c0, c1) pairs, each 1/30; diagonal (same card) 0.
    j = PBS.initial().joint_belief()
    assert j.shape == (NUM_CARDS, NUM_CARDS)
    assert np.isclose(j.sum(), 1.0)
    off = ~np.eye(NUM_CARDS, dtype=bool)
    assert np.allclose(j[off], 1.0 / 30.0)
    assert np.allclose(np.diag(j), 0.0)


def test_initial_marginals_are_uniform_sixths():
    pbs = PBS.initial()
    assert np.allclose(pbs.marginal(0), 1.0 / 6.0)
    assert np.allclose(pbs.marginal(1), 1.0 / 6.0)
    assert np.isclose(pbs.marginal(0).sum(), 1.0)


def test_public_card_blocks_that_rank_index_from_both_ranges():
    # After a public card is set, no player can hold that exact card.
    pbs = PBS.initial()
    pbs.public_card = 4  # a King
    j = pbs.joint_belief()
    assert np.isclose(j.sum(), 1.0)
    assert np.allclose(j[4, :], 0.0)
    assert np.allclose(j[:, 4], 0.0)
    assert np.isclose(pbs.marginal(0)[4], 0.0)


def test_bayes_update_with_separating_strategy_concentrates_range():
    # P0 plays a pure separating strategy: RAISE iff holding a King, else CALL.
    # action_probs[c] = prob P0 takes the observed action given card c.
    pbs = PBS.initial()
    raise_probs = np.array([0, 0, 0, 0, 1, 1], dtype=float)  # 1 for K = {4,5}
    after_raise = pbs.update(player=0, action_probs=raise_probs)
    # P0's marginal now sits entirely on the two Kings, 1/2 each.
    m = after_raise.marginal(0)
    assert np.allclose(m, [0, 0, 0, 0, 0.5, 0.5])
    # Opponent (P1) marginal stays informative but excludes blocking with P0's Ks.
    assert np.isclose(after_raise.marginal(1).sum(), 1.0)

    call_probs = np.array([1, 1, 1, 1, 0, 0], dtype=float)  # non-K
    after_call = pbs.update(player=0, action_probs=call_probs)
    assert np.allclose(after_call.marginal(0), [0.25, 0.25, 0.25, 0.25, 0, 0])


def test_update_is_pure_bayes_multiplication_on_reach():
    pbs = PBS.initial()
    probs = np.array([0.2, 0.2, 0.5, 0.5, 0.9, 0.9], dtype=float)
    out = pbs.update(player=1, action_probs=probs)
    assert np.allclose(out.reach[1], probs)          # reach[1] *= probs
    assert np.allclose(out.reach[0], 1.0)            # other player untouched
    assert pbs.reach[1].tolist() == [1.0] * 6        # original unmutated


def test_feature_vector_has_documented_shape():
    feats = PBS.initial().features()
    # 5 one-hot betting lines + 2*6 reach entries = 17.
    assert feats.shape == (17,)
