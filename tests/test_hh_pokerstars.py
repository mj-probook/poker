"""Slice F: PokerStars tournament HH parser → engine replay validation.

Each authored fixture is parsed to a `HandSetup` + engine action list, then
replayed through the Slice-A `Hand` engine. The engine independently computes
side pots / uncalled returns / showdown distribution; we assert its final
stacks and contested pot match the results *stated in the hand history text*.
"""

from pathlib import Path

from pokerlab.engine.state import Hand
from pokerlab.hh.pokerstars import parse_pokerstars

FIXTURES = Path(__file__).parent / "fixtures" / "hh"


def _replay(parsed):
    return Hand.replay(parsed.setup, parsed.actions)


def _assert_replay_matches(parsed) -> None:
    hand = _replay(parsed)
    assert hand.is_terminal()
    assert tuple(hand.final_stacks()) == parsed.stated_final_stacks()
    assert sum(hand.contrib) - parsed.uncalled == parsed.total_pot


def test_preflop_fold_replays_to_stated_stacks() -> None:
    parsed = parse_pokerstars((FIXTURES / "ps_preflop_fold.txt").read_text())
    assert parsed.site == "PokerStars"
    assert parsed.setup.bb == 100 and parsed.setup.sb == 50 and parsed.setup.ante == 10
    # 6 seated, no board dealt (folds preflop)
    assert len(parsed.setup.stacks) == 6
    assert parsed.setup.board == ()
    _assert_replay_matches(parsed)


def test_multiway_flop_showdown_replays() -> None:
    parsed = parse_pokerstars((FIXTURES / "ps_multiway_flop.txt").read_text())
    assert parsed.setup.ante == 0  # this fixture exercises the no-ante path
    assert len(parsed.setup.board) == 5
    _assert_replay_matches(parsed)


def test_allin_sidepot_replays() -> None:
    parsed = parse_pokerstars((FIXTURES / "ps_allin_sidepot.txt").read_text())
    # 3-way all-in -> main + side pot; short stack collects only the main pot
    assert parsed.total_pot == 7000 and parsed.uncalled == 2000
    _assert_replay_matches(parsed)


def test_ante_heads_up_showdown_replays() -> None:
    parsed = parse_pokerstars((FIXTURES / "ps_ante_hu.txt").read_text())
    assert parsed.setup.ante == 75 and len(parsed.setup.stacks) == 2
    _assert_replay_matches(parsed)
