"""GGPoker (Pokercraft) tournament hand-history parser (Slice F, impl doc §3).

Same body grammar as PokerStars (see `hh._common`); differences handled here:
  * header reads ``Poker Hand #...: Tournament #..., <name> Hold'em No Limit -
    Level<n>(sb/bb[/ante]) - <date>`` (no roman numerals, no space before the
    level number, optional ante field in the parens);
  * villains are anonymized hex-ish ids, tolerated by the shared name regexes;
  * SUMMARY carries extra ``| Jackpot | Bingo`` fields and rake may be absent —
    the ``Total pot`` search ignores everything after the amount.
"""

from __future__ import annotations

import re

from pokerlab.hh._common import parse_hand, peek_hand_uid, split_hands
from pokerlab.hh.model import ParsedHand

# See pokerstars.BOUNDARY. GGPoker shares `_common.parse_hand`, so it shared the
# single-hand defect (round-4 finding [R2']) and is fixed on the same code path
# rather than deferred — Pokercraft exports are session files too.
#
# No collision with PokerStars' prefix despite the shared word: "PokerStars Hand
# #" does not start with "Poker Hand #". `detect_site` routes on the same two
# literals, so a file reaching this parser splits on this boundary by
# construction.
BOUNDARY = "Poker Hand #"

_HEADER = re.compile(
    r"Poker Hand #(?P<hid>\w+): Tournament #(?P<tid>\d+),.*?"
    r"- Level\s*(?P<level>[\w-]+)\s*\((?P<sb>\d+)/(?P<bb>\d+)(?:/\d+)?\)"
)


def parse_ggpoker(text: str) -> ParsedHand:
    """Parse exactly ONE hand. Raises if handed a multi-hand session file."""
    return parse_hand(text, site="GGPoker", header_re=_HEADER,
                      boundary=BOUNDARY)


def split_ggpoker(text: str) -> list[str]:
    """Session file -> one raw chunk per hand (no parsing, no validation)."""
    return split_hands(text, BOUNDARY)


def parse_ggpoker_file(text: str) -> list[ParsedHand]:
    """Every hand in a session file. See `pokerstars.parse_pokerstars_file`."""
    return [parse_ggpoker(c) for c in split_ggpoker(text)]


def peek_ggpoker_uid(text: str) -> str | None:
    """The hand number from this chunk's header, or None if it won't parse.

    For the failure path: a body-level failure still has a real hand number and
    should be recorded under it, not as "unidentified" ([R1b]).
    """
    return peek_hand_uid(text, _HEADER)
