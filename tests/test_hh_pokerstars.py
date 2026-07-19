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


# --------------------------------------------------------------------------- #
# Round-1 finding [9]: the dead button. When a player busts, the announced
# button seat can be an EMPTY seat — the button is "dead" and the blinds stay
# where the rotation put them. Parsing must not crash on it.
# --------------------------------------------------------------------------- #
def test_dead_button_falls_back_to_an_occupied_seat() -> None:
    parsed = parse_pokerstars((FIXTURES / "ps_dead_button.txt").read_text())

    # seats {1, 2, 5} occupied; the announced button (seat #4) is empty
    assert len(parsed.setup.stacks) == 3
    # nearest occupied seat counter-clockwise from #4 is #2 -> engine index 1
    assert parsed.setup.button == 1
    _assert_replay_matches(parsed)


def test_dead_button_fallback_agrees_with_the_posted_blinds() -> None:
    """The fallback is only correct if it reproduces the blinds the HH states.

    Engine order for n>=3 is SB=(button+1)%n, BB=(button+2)%n; the fixture posts
    the SB from seat #5 and the BB from seat #1, which pins the button to #2.
    """
    from pokerlab.hh.decisions import position_label

    parsed = parse_pokerstars((FIXTURES / "ps_dead_button.txt").read_text())
    n, button = len(parsed.setup.stacks), parsed.setup.button
    names = parsed.seat_names
    pos = {names[i]: position_label(n, button, i) for i in range(n)}

    assert pos["VillainB"] == "SB"   # seat 5 posted the small blind
    assert pos["Hero"] == "BB"       # seat 1 posted the big blind
    assert pos["VillainA"] == "BTN"  # seat 2 carries the dead button
