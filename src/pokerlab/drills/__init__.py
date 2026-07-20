"""Drill engine (M2; plan §5.1, §5.3; impl doc §3 Slice E).

Turns the in-house chart engine into a scored, spaced-repetition training loop.
Public surface consumed by the web UI (and, later, Slice F's HH grading):

    scoring.score(solution, chosen, pot_bb) -> Score   # decision-ε rule
    generator.default_population()          -> [Drill]
    scheduler.schedule_attempt / select_next            # SM-2
    agents.chart_argmax_agent / run_agent               # M2 exit proof
"""

from .scoring import score

__all__ = ["score"]
