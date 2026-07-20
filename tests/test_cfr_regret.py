"""Wave-2 [Q4]: the shared regret-matching rule.

The rule was duplicated across four solvers. These pin the contract once, and
— the part that actually matters — pin that the pure-Python and vectorized
implementations agree, since they exist as two functions only for speed.
"""

import numpy as np
import pytest

from pokerlab.cfr.regret import regret_match, regret_match_np


def test_strategy_is_proportional_to_positive_regret():
    assert regret_match([3.0, 1.0]) == pytest.approx([0.75, 0.25])


def test_negative_regret_is_clipped_not_carried():
    # the negative action gets zero weight; the rest split the mass
    assert regret_match([-5.0, 1.0, 3.0]) == pytest.approx([0.0, 0.25, 0.75])


def test_uniform_when_no_regret_is_positive():
    """t=0 (all-zero regret) and the all-negative case both fall back uniform."""
    assert regret_match([0.0, 0.0, 0.0]) == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert regret_match([-1.0, -2.0]) == pytest.approx([0.5, 0.5])


def test_strategy_is_a_distribution():
    for r in ([3.0, 1.0], [-5.0, 1.0, 3.0], [0.0, 0.0], [-1.0, -1.0, -1.0]):
        assert sum(regret_match(r)) == pytest.approx(1.0)


@pytest.mark.parametrize("axis", [0, 1])
def test_numpy_matches_pure_python_on_every_slice(axis):
    """The two implementations exist only for speed — they must not diverge."""
    rng = np.random.default_rng(7)
    reg = rng.normal(size=(4, 5)) * 2.0
    reg[:, 0] = -1.0          # a fully-negative slice (uniform fallback)
    reg[:, 1] = 0.0           # an all-zero slice (t=0)

    got = regret_match_np(reg, axis=axis)
    # compare against the scalar rule applied to each 1-D slice along `axis`
    moved = np.moveaxis(reg, axis, -1)
    want = np.moveaxis(
        np.array([regret_match(list(v)) for v in moved.reshape(-1, moved.shape[-1])]
                 ).reshape(moved.shape), -1, axis)
    assert got == pytest.approx(want)
    assert got.sum(axis=axis) == pytest.approx(np.ones(got.sum(axis=axis).shape))


def test_the_two_implementations_agree_on_nan():
    """[M12] They used to disagree: maximum() propagates NaN and sends the whole
    infoset uniform, while the scalar comparison drops just that action.

    NaN regret means something upstream is already broken, but the two solvers
    must not disagree about what a broken input means — that is the property
    the shared module exists to provide.
    """
    reg = [float("nan"), 1.0, -2.0]
    want = regret_match(reg)
    got = regret_match_np(np.array(reg).reshape(-1, 1), axis=0).ravel()
    assert got == pytest.approx(want)
    assert want == pytest.approx([0.0, 1.0, 0.0])
