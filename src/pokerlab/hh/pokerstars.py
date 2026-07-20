"""PokerStars tournament hand-history parser (Slice F, impl doc §3).

Thin wrapper over `hh._common.parse_hand`: only the header line is
site-specific ("PokerStars Hand #... - Level V (50/100)").
"""

from __future__ import annotations

import re

from pokerlab.hh._common import parse_hand, peek_hand_uid, split_hands
from pokerlab.hh.model import ParsedHand

# The hand-header prefix, used to split session files into per-hand chunks.
# Deliberately looser than `_HEADER`: a hand with a malformed header must still
# start its own chunk so it can be isolated and recorded, rather than glued
# onto its predecessor (round-4 finding [R2']). Same literal `detect_site`
# sniffs for, and the two must agree — a file we route to this parser is a file
# we can split.
BOUNDARY = "PokerStars Hand #"

_HEADER = re.compile(
    r"PokerStars Hand #(?P<hid>\d+): Tournament #(?P<tid>\d+),.*?"
    r"- Level (?P<level>[\w-]+) \((?P<sb>\d+)/(?P<bb>\d+)(?:/\d+)?\)"
)


def parse_pokerstars(text: str) -> ParsedHand:
    """Parse exactly ONE hand. Raises if handed a multi-hand session file."""
    return parse_hand(text, site="PokerStars", header_re=_HEADER,
                      boundary=BOUNDARY)


def split_pokerstars(text: str) -> list[str]:
    """Session file -> one raw chunk per hand (no parsing, no validation)."""
    return split_hands(text, BOUNDARY)


def parse_pokerstars_file(text: str) -> list[ParsedHand]:
    """Every hand in a session file (plan §5.3: auto-saved local files).

    Raises on the first unparseable chunk. Callers that must not lose the good
    hands to one bad one — the CLI import path — split with `split_pokerstars`
    and isolate per chunk themselves; that machinery already exists and this
    does not duplicate it.
    """
    return [parse_pokerstars(c) for c in split_pokerstars(text)]


def peek_pokerstars_uid(text: str) -> str | None:
    """The hand number from this chunk's header, or None if it won't parse.

    For the failure path: a body-level failure still has a real hand number and
    should be recorded under it, not as "unidentified" ([R1b]).
    """
    return peek_hand_uid(text, _HEADER)
