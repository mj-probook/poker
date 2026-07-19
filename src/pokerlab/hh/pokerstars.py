"""PokerStars tournament hand-history parser (Slice F, impl doc §3).

Thin wrapper over `hh._common.parse_hand`: only the header line is
site-specific ("PokerStars Hand #... - Level V (50/100)").
"""

from __future__ import annotations

import re

from pokerlab.hh._common import parse_hand
from pokerlab.hh.model import ParsedHand

_HEADER = re.compile(
    r"PokerStars Hand #(?P<hid>\d+): Tournament #(?P<tid>\d+),.*?"
    r"- Level (?P<level>[\w-]+) \((?P<sb>\d+)/(?P<bb>\d+)(?:/\d+)?\)"
)


def parse_pokerstars(text: str) -> ParsedHand:
    return parse_hand(text, site="PokerStars", header_re=_HEADER)
