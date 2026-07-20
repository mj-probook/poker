"""169 preflop hand-class taxonomy (impl doc §3 Slice C).

The shared vocabulary for the chart engine: canonical class labels ("AA",
"AKs", "AKo", ...), their combo counts, and the concrete 2-card combinations
each class expands to under the frozen `types.Card` codec
(card = (rank-2)*4 + suit; rank 2..14, suit 0..3).

Canonical order = row-major over the 13×13 rank grid with ranks descending
A..2: the diagonal is pairs, the upper triangle (higher rank first) is suited,
the lower triangle is offsuit. Deterministic and stable — the equity matrix and
every chart index against it.
"""

from __future__ import annotations

# Index 0..12 -> rank value 14..2 (A high).
RANK_CHARS = "AKQJT98765432"
_CHAR_TO_RANK = {c: 14 - i for i, c in enumerate(RANK_CHARS)}


def _build_classes() -> list[str]:
    classes: list[str] = []
    for i, hi in enumerate(RANK_CHARS):
        for j, lo in enumerate(RANK_CHARS):
            if i == j:
                classes.append(hi + hi)          # pair, e.g. "AA"
            elif i < j:
                classes.append(hi + lo + "s")     # suited, higher rank first
            else:
                classes.append(lo + hi + "o")     # offsuit, higher rank first
    return classes


HAND_CLASSES: list[str] = _build_classes()
HAND_INDEX: dict[str, int] = {h: i for i, h in enumerate(HAND_CLASSES)}


def is_pair(label: str) -> bool:
    return len(label) == 2 and label[0] == label[1]


def hand_ranks(label: str) -> tuple[int, int]:
    """(high_rank, low_rank) as rank values 2..14, high >= low."""
    if is_pair(label):
        r = _CHAR_TO_RANK[label[0]]
        return (r, r)
    return (_CHAR_TO_RANK[label[0]], _CHAR_TO_RANK[label[1]])


def combos(label: str) -> int:
    """Number of concrete 2-card combinations: pair 6, suited 4, offsuit 12."""
    if is_pair(label):
        return 6
    return 4 if label.endswith("s") else 12


def _rank_index(rank_value: int) -> int:
    return rank_value - 2


def card_combos(label: str) -> list[tuple[int, int]]:
    """Every concrete (card, card) pair the class expands to (frozen codec)."""
    hi, lo = hand_ranks(label)
    hi_i, lo_i = _rank_index(hi), _rank_index(lo)
    out: list[tuple[int, int]] = []
    if is_pair(label):
        for s1 in range(4):
            for s2 in range(s1 + 1, 4):
                out.append((hi_i * 4 + s1, hi_i * 4 + s2))
    elif label.endswith("s"):
        for s in range(4):
            out.append((hi_i * 4 + s, lo_i * 4 + s))
    else:  # offsuit: distinct suits
        for s1 in range(4):
            for s2 in range(4):
                if s1 != s2:
                    out.append((hi_i * 4 + s1, lo_i * 4 + s2))
    return out
