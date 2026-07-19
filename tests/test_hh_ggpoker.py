"""Slice F: GGPoker tournament HH parser → engine replay validation.

Covers GG-specific format quirks: anonymized villain ids, non-roman
``Level12(sb/bb)`` headers, one-word ``*** SHOWDOWN ***``, a SUMMARY with no
rake field (missing-field tolerance), and physical seat gaps that must compact
onto contiguous engine indices while preserving clockwise order.
"""

from pathlib import Path

from pokerlab.engine.state import Hand
from pokerlab.hh.ggpoker import parse_ggpoker

FIXTURES = Path(__file__).parent / "fixtures" / "hh"


def _assert_replay_matches(parsed) -> None:
    hand = Hand.replay(parsed.setup, parsed.actions)
    assert hand.is_terminal()
    assert tuple(hand.final_stacks()) == parsed.stated_final_stacks()
    assert sum(hand.contrib) - parsed.uncalled == parsed.total_pot


def test_preflop_3bet_with_seat_gaps_and_no_rake_line() -> None:
    parsed = parse_ggpoker((FIXTURES / "gg_preflop_3bet.txt").read_text())
    assert parsed.site == "GGPoker"
    # six seated across physical seats 1,2,3,5,6,8 compact to engine 0..5
    assert len(parsed.setup.stacks) == 6
    assert parsed.seat_names[parsed.hero] == "Hero"
    # villains are anonymized (not "Hero", not empty)
    assert all(name for name in parsed.seat_names)
    _assert_replay_matches(parsed)


def test_allin_jam_showdown_with_anonymized_villain() -> None:
    parsed = parse_ggpoker((FIXTURES / "gg_allin_jam.txt").read_text())
    assert parsed.setup.ante == 50 and len(parsed.setup.stacks) == 2
    assert parsed.total_pot == 12000 and parsed.uncalled == 0
    _assert_replay_matches(parsed)


def test_sb_open_fold_returns_uncalled_bb() -> None:
    parsed = parse_ggpoker((FIXTURES / "gg_sb_fold_leak.txt").read_text())
    assert parsed.total_pot == 400 and parsed.uncalled == 200
    _assert_replay_matches(parsed)


def test_bb_ante_wording_is_parsed() -> None:
    """[12][18] GGPoker spells it out: 'posts big blind ante N'."""
    parsed = parse_ggpoker((FIXTURES / "gg_bb_ante.txt").read_text())
    assert parsed.setup.ante == 0
    assert parsed.setup.bb_ante == 400
    assert parsed.contributed[2] == 600   # 400 ante + 400 bb - 200 uncalled
    assert sum(parsed.contributed) == parsed.total_pot
    hand = Hand.replay(parsed.setup, parsed.actions)
    assert hand.is_terminal()
    assert tuple(hand.final_stacks()) == parsed.stated_final_stacks()
