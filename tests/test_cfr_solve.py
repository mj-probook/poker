"""Slice B — CFR / CFR+ convergence, judged by the (already-verified) instrument.

Kuhn here; Leduc lives in test_cfr_leduc.py. MCCFR in test_cfr_mccfr.py.
"""

import math

from pokerlab.cfr.cfr import CFRSolver
from pokerlab.cfr.exploit import nash_conv, on_policy_values
from pokerlab.cfr.kuhn import KuhnPoker


def test_vanilla_cfr_nash_conv_trends_down():
    s = CFRSolver(KuhnPoker(), plus=False)
    marks = []
    for t in (10, 100, 1000):
        s.run(t - s.t)
        marks.append(nash_conv(s.tree, s.average_profile()))
    assert marks[0] > marks[1] > marks[2]
    assert marks[2] < 5e-3  # vanilla is slow but must be clearly converging


def test_cfr_plus_kuhn_reaches_1e_3():
    s = CFRSolver(KuhnPoker(), plus=True)
    s.run(500)  # crosses 1e-3 near ~200 iters; 500 is a safe CI budget (<<30s)
    assert nash_conv(s.tree, s.average_profile()) <= 1e-3


def test_cfr_plus_kuhn_recovers_game_value():
    s = CFRSolver(KuhnPoker(), plus=True)
    s.run(500)
    p0_value = on_policy_values(s.tree, s.average_profile())[0]
    assert math.isclose(p0_value, -1.0 / 18.0, abs_tol=1e-3)
