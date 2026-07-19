"""Contract smoke: the frozen shared types are importable and constructible."""

from pokerlab.types import (
    TIER_BEST_AVAILABLE,
    GameState,
    Score,
    Solution,
    SpotKey,
    TournamentContext,
)


def test_contract_types_construct() -> None:
    tc = TournamentContext(payouts=(500000, 300000, 200000), players_remaining=3,
                           stacks_all=(4000, 3000, 2000), bb=200, ante=25)
    gs = GameState(seats=3, stacks=[4000, 3000, 2000], button=0, tournament=tc)
    key = SpotKey(formation="BTNopen_BBcall", stack_bucket=40, board_bucket="ish")
    sol = Solution(actions={"jam": (0.42, 0.97), "fold": (0.0, 0.03)},
                   range_ctx="test", source="chart")
    score = Score(correct=True, ev_loss_bb=0.0, best_action="jam", chosen_frequency=0.97)
    assert gs.street == "preflop" and gs.tournament is tc
    assert key.stack_bucket == 40
    assert sol.actions["jam"][1] > 0.9
    assert score.correct and TIER_BEST_AVAILABLE == 3
