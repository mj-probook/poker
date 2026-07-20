"""Reference drill agents + harness (Slice E; M2 exit proof).

The M2 milestone exit is a *consistency proof* of the scoring rule and the
generator: an agent that always plays the chart's argmax action must clear the
decision-ε rule on ≥98% of a mixed drill run. A deliberately bad agent (always
fold) must score far lower — evidence the metric can actually fail.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pokerlab.drills.generator import Drill
from pokerlab.drills.scoring import score

Agent = Callable[[Drill], str]


def chart_argmax_agent(drill: Drill) -> str:
    """Play the chart's most-frequent (recommended) action."""
    acts = drill.solution.actions
    return max(acts, key=lambda a: acts[a][1])


def always_fold_agent(drill: Drill) -> str:
    """Deliberately bad baseline: fold everything."""
    return "fold"


@dataclass(frozen=True)
class AgentResult:
    n: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0


def run_agent(agent: Agent, drills: list[Drill]) -> AgentResult:
    """Play `agent` through every drill and score against its chart Solution."""
    correct = 0
    for d in drills:
        chosen = agent(d)
        if chosen not in d.solution.actions:
            raise ValueError(f"agent returned illegal action {chosen!r} "
                             f"for {d.drill_id}")
        if score(d.solution, chosen, d.pot_bb).correct:
            correct += 1
    return AgentResult(n=len(drills), correct=correct)
