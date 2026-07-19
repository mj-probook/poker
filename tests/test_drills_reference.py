"""Slice E — M2 exit proof: reference chart-argmax agent ≥98% (plan §8 M2).

This is the milestone gate. It proves the scoring rule and the generator are
mutually consistent: playing the chart's own recommendation clears the
decision-ε rule almost everywhere, while an always-fold agent scores far lower
(so the accuracy metric can genuinely fail).
"""

from pokerlab.drills import generator as gen
from pokerlab.drills.agents import always_fold_agent, chart_argmax_agent, run_agent


def _mixed_500():
    return gen.sample_drills(gen.jamfold_drills(), 500, seed=0)


def test_reference_agent_clears_98pct_over_500_mixed_drills():
    res = run_agent(chart_argmax_agent, _mixed_500())
    assert res.n == 500
    assert res.accuracy >= 0.98, res.accuracy


def test_always_fold_agent_scores_far_lower():
    drills = _mixed_500()
    ref = run_agent(chart_argmax_agent, drills).accuracy
    fold = run_agent(always_fold_agent, drills).accuracy
    assert fold < 0.75                 # folding is wrong on huge swaths
    assert ref - fold >= 0.25          # the metric clearly separates them


def test_reference_agent_perfect_on_full_chip_population():
    # Over the entire syllabus (not just a sample) the chart argmax is exact.
    res = run_agent(chart_argmax_agent, gen.jamfold_drills())
    assert res.accuracy >= 0.98, res.accuracy
