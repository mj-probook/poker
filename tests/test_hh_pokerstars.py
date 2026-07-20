"""Slice F: PokerStars tournament HH parser → engine replay validation.

Each authored fixture is parsed to a `HandSetup` + engine action list, then
replayed through the Slice-A `Hand` engine. The engine independently computes
side pots / uncalled returns / showdown distribution; we assert its final
stacks and contested pot match the results *stated in the hand history text*.
"""

from pathlib import Path

import pytest

from pokerlab.engine.state import Hand
from pokerlab.hh.decisions import reconcile
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


# --------------------------------------------------------------------------- #
# Round-1 finding [12][18]: the BB ante. Modern tournaments (the target format)
# have the big blind post ONE ante for the table. The parser read the first
# "posts the ante N" line as a PER-PLAYER ante, so the engine charged every
# seat — an n-fold overcharge — and never matched "posts big blind ante" at all.
# --------------------------------------------------------------------------- #
def test_bb_ante_is_not_charged_to_every_seat() -> None:
    parsed = parse_pokerstars((FIXTURES / "ps_bb_ante.txt").read_text())

    # a single ante line in a 6-handed hand is a BB ante, not a per-player one
    assert parsed.setup.ante == 0
    assert parsed.setup.bb_ante == 600
    # only the big blind paid it
    assert parsed.contributed[2] == 900   # 600 ante + 600 bb - 300 uncalled
    assert sum(parsed.contributed) == parsed.total_pot
    _assert_replay_matches(parsed)


def test_per_player_ante_fixture_is_unchanged() -> None:
    """The per-player path must be untouched (two ante lines, HU)."""
    parsed = parse_pokerstars((FIXTURES / "ps_ante_hu.txt").read_text())
    assert parsed.setup.ante == 75 and parsed.setup.bb_ante == 0
    _assert_replay_matches(parsed)


# --------------------------------------------------------------------------- #
# Round-2 finding [E6]: the ante classifier keyed off HOW MANY seats posted an
# ante rather than WHO did, and excluded n==2 outright. Modern PokerStars emits
# a heads-up big-blind ante as a lone unlabelled "posts the ante" line, so both
# seats were charged and every such hand died at reconciliation -- HU at <=20bb
# is exactly the tier-1 jam/fold class the pipeline exists for. The rule is now
# "a lone ante posted by the big blind is a BB ante", which needs no seat-count
# special case and settles the 3-handed case below in the other direction.
# --------------------------------------------------------------------------- #
def test_heads_up_big_blind_ante_is_not_charged_to_both_seats() -> None:
    parsed = parse_pokerstars((FIXTURES / "ps_bb_ante_hu.txt").read_text())
    assert parsed.setup.bb_ante == 600
    assert parsed.setup.ante == 0          # was 600 -> both seats overcharged
    _assert_replay_matches(parsed)


def test_lone_per_player_ante_from_a_non_blind_stays_per_player() -> None:
    """3-handed, one genuine per-player ante line: the poster is NOT the BB."""
    text = (
        "PokerStars Hand #240000000012: Tournament #3900000012, $100+$9 USD "
        "Hold'em No Limit - Level XX (100/200) - 2024/03/06 15:00:00 ET\n"
        "Table '3900000012 3' 3-max Seat #1 is the button\n"
        "Seat 1: VillainA (5000 in chips)\n"
        "Seat 2: VillainB (5000 in chips)\n"
        "Seat 3: Hero (5000 in chips)\n"
        "VillainA: posts the ante 25\n"
        "VillainB: posts small blind 100\n"
        "Hero: posts big blind 200\n"
        "*** HOLE CARDS ***\n"
        "Dealt to Hero [As Ks]\n"
        "VillainA: folds\n"
        "VillainB: folds\n"
        "Uncalled bet (100) returned to Hero\n"
        "Hero collected 225 from pot\n"
        "*** SUMMARY ***\n"
        "Total pot 225 | Rake 0\n"
        "Seat 3: Hero (big blind) collected (225)\n"
    )
    parsed = parse_pokerstars(text)
    assert parsed.setup.ante == 25         # was read as a BB ante
    assert parsed.setup.bb_ante == 0


def test_multiway_big_blind_ante_fixture_is_unchanged() -> None:
    """The lone poster IS the big blind here, so this stays a BB ante."""
    parsed = parse_pokerstars((FIXTURES / "ps_bb_ante.txt").read_text())
    assert parsed.setup.bb_ante == 600 and parsed.setup.ante == 0
    _assert_replay_matches(parsed)


# --------------------------------------------------------------------------- #
# Round-3 findings [H1][H5]: the ante amount and the ante STRUCTURE are two
# separate readings, and both were fragile. H1: the amount came from the first
# ante line, which is the *shortfall* when a short stack posts partially and
# sits in an earlier seat. H5: the structure is textually ambiguous when the
# file does not name the big blind as the poster, so the stated result arbitrates.
# --------------------------------------------------------------------------- #
def test_partial_ante_from_a_short_stack_does_not_set_the_table_ante() -> None:
    """[H1] a player too short to pay posts a partial ante and is all-in.

    The ante is what everyone OWES, so it is the maximum posted, not the first
    line. The engine already clamps each seat to its stack, so passing the full
    ante is also what makes the short stack come out right.
    """
    text = (
        "PokerStars Hand #240000000032: Tournament #3900000032, $10+$1 USD "
        "Hold'em No Limit - Level X (100/200) - 2024/03/01 20:15:00 ET\n"
        "Table '3900000032 7' 9-max Seat #1 is the button\n"
        "Seat 1: Villain1 (25 in chips)\n"
        "Seat 2: Villain2 (5000 in chips)\n"
        "Seat 3: Hero (5000 in chips)\n"
        "Villain1: posts the ante 25 and is all-in\n"   # the SHORTFALL, listed first
        "Villain2: posts the ante 50\n"
        "Hero: posts the ante 50\n"
        "Villain2: posts small blind 100\n"
        "Hero: posts big blind 200\n"
        "*** HOLE CARDS ***\n"
        "Dealt to Hero [Ah Kd]\n"
        "Villain2: folds\n"
        "Uncalled bet (100) returned to Hero\n"
        "Hero collected 325 from pot\n"
        "*** SUMMARY ***\n"
        "Total pot 325 | Rake 0\n"
    )
    assert parse_pokerstars(text).setup.ante == 50      # was 25 -> undercharged


@pytest.mark.parametrize("fixture,correct", [("ps_bb_ante.txt", "bb_ante"),
                                             ("ps_ante_hu.txt", "ante")])
def test_stated_result_arbitrates_an_ambiguous_ante_reading(fixture, correct) -> None:
    """[H5] feed the arbitration the WRONG structure; it must recover it.

    Tested at the mechanism rather than through a contrived fixture: the
    identity rule already reads every checked-in hand correctly, so a fixture
    that exercised the fallback would have to be one we do not believe in.
    Both directions are covered, and a correct reading must never be moved.
    """
    import dataclasses

    from pokerlab.hh._common import _arbitrate_ante

    parsed = parse_pokerstars((FIXTURES / fixture).read_text())
    amount = parsed.setup.ante or parsed.setup.bb_ante
    kwargs = dict(actions=parsed.actions, stated_pot=parsed.total_pot,
                  uncalled=parsed.uncalled,
                  stated_final=parsed.stated_final_stacks())

    wrong = dataclasses.replace(
        parsed.setup,
        ante=0 if parsed.setup.ante else amount,
        bb_ante=amount if parsed.setup.ante else 0)
    recovered = _arbitrate_ante(wrong, **kwargs)
    assert ("bb_ante" if recovered.bb_ante else "ante") == correct

    # ...and an already-correct reading is left exactly alone
    assert _arbitrate_ante(parsed.setup, **kwargs) == parsed.setup


def test_arbitration_keeps_the_original_when_neither_reading_reconciles() -> None:
    """A hand we cannot model must be DROPPED loudly, not silently reshaped.

    One non-blind seat posting a lone ante is neither a per-player ante (the
    others did not post) nor a BB ante (the poster is not the BB), so both
    readings fail and reconciliation is the correct place for it to die.
    """
    text = (
        "PokerStars Hand #240000000031: Tournament #3900000031, $10+$1 USD "
        "Hold'em No Limit - Level X (100/200) - 2024/03/01 20:15:00 ET\n"
        "Table '3900000031 7' 9-max Seat #1 is the button\n"
        "Seat 1: Villain1 (5000 in chips)\n"
        "Seat 2: Villain2 (5000 in chips)\n"
        "Seat 3: Hero (5000 in chips)\n"
        "Villain1: posts the ante 25\n"
        "Villain2: posts small blind 100\n"
        "Hero: posts big blind 200\n"
        "*** HOLE CARDS ***\n"
        "Dealt to Hero [Ah Kd]\n"
        "Villain1: folds\n"
        "Villain2: folds\n"
        "Uncalled bet (100) returned to Hero\n"
        "Hero collected 325 from pot\n"
        "*** SUMMARY ***\n"
        "Total pot 325 | Rake 0\n"
    )
    parsed = parse_pokerstars(text)
    assert parsed.setup.ante == 25 and parsed.setup.bb_ante == 0   # unmoved
    with pytest.raises(ValueError, match="reconciliation failed"):
        reconcile(parsed)
