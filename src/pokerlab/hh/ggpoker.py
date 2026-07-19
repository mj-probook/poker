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

from pokerlab.hh._common import parse_hand
from pokerlab.hh.model import ParsedHand

_HEADER = re.compile(
    r"Poker Hand #(?P<hid>\w+): Tournament #(?P<tid>\d+),.*?"
    r"- Level\s*(?P<level>[\w-]+)\s*\((?P<sb>\d+)/(?P<bb>\d+)(?:/\d+)?\)"
)


def parse_ggpoker(text: str) -> ParsedHand:
    return parse_hand(text, site="GGPoker", header_re=_HEADER)
