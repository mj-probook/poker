"""8-way flop texture taxonomy (plan §1 SpotKey; impl doc §3 Slice D).

Every flop is labelled by exactly one of 8 classes. The raw feature space is
``paired? × {monotone, two_tone, rainbow} × {connected, dry}`` = 2·3·2 = 12,
collapsed to 8 by two structural facts:

  * A **paired** flop can never be monotone (its pair already spans two suits),
    so paired suitedness ∈ {two_tone, rainbow} — that removes the paired-mono
    combinations outright.
  * For paired flops the strategic character is dominated by the pair and the
    flush threat; straightness is secondary, so **connectedness is folded away
    for paired boards**.

The resulting 8 labels (``TEXTURES``):

  unpaired (6):  mono_conn   mono_dry
                 twotone_conn twotone_dry
                 rainbow_conn rainbow_dry
  paired   (2):  paired_twotone  paired_rainbow   (trips → paired_rainbow)

Suitedness = count of the most common suit among the 3 cards: 3→monotone,
2→two_tone, 1→rainbow. Connectedness (unpaired only): the three distinct ranks
are ``connected`` iff they fit inside one 5-card straight window, i.e.
``max-min ≤ 4`` under the better of ace-high(=14) / ace-low(=1); else ``dry``.

Every feature is suit-relabelling invariant, so the label of a flop equals the
label of its canonical form (solver/iso.py).
"""

from __future__ import annotations

from collections import Counter

from pokerlab.engine.cards import card_rank, card_suit
from pokerlab.types import Card

TEXTURES: tuple[str, ...] = (
    "mono_conn",
    "mono_dry",
    "twotone_conn",
    "twotone_dry",
    "rainbow_conn",
    "rainbow_dry",
    "paired_twotone",
    "paired_rainbow",
)

_SUITEDNESS = {3: "mono", 2: "twotone", 1: "rainbow"}


def _is_connected(ranks: list[int]) -> bool:
    """Three *distinct* ranks fit inside one 5-card straight window."""
    span_high = max(ranks) - min(ranks)
    if 14 in ranks:  # allow the wheel: treat the ace as low
        low = [1 if r == 14 else r for r in ranks]
        span_low = max(low) - min(low)
        return min(span_high, span_low) <= 4
    return span_high <= 4


def texture(flop: tuple[Card, ...]) -> str:
    """One of the 8 ``TEXTURES`` labels for a 3-card flop."""
    if len(flop) != 3:
        raise ValueError(f"texture expects a 3-card flop, got {len(flop)}")
    ranks = [card_rank(c) for c in flop]
    suits = [card_suit(c) for c in flop]
    top_suit_count = max(Counter(suits).values())
    suitedness = _SUITEDNESS[top_suit_count]
    paired = len(set(ranks)) < 3

    if paired:
        # monotone impossible when paired; only two_tone / rainbow remain.
        return "paired_twotone" if suitedness == "twotone" else "paired_rainbow"
    conn = "conn" if _is_connected(ranks) else "dry"
    return f"{suitedness}_{conn}"
