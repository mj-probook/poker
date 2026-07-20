"""Hygiene guard on the hand-history fixtures themselves.

Dedup is keyed on `(site, hand_uid)`, so two DIFFERENT fixture hands sharing a
site hand number is not a cosmetic wart: importing both in one session silently
drops one, with no error and no failed_hands row — just a count one short. That
is the worst shape a fixture bug can take, because the test written on top of it
fails while describing something that is not wrong.

The guard has to distinguish two cases that both look like a repeated uid:

  * LEGITIMATE — the multi-hand session fixtures ([R2']) embed copies of the
    single-hand fixtures, so the same hand text appears in several files. That
    is the point of those fixtures.
  * A COLLISION — two hands with DIFFERENT text claiming the same number.

So the assertion is per-uid: every occurrence must be the same hand.
"""

from pathlib import Path

from pokerlab.hh.ggpoker import split_ggpoker
from pokerlab.hh.pokerstars import split_pokerstars

FIXTURES = Path(__file__).parent / "fixtures" / "hh"


def _hands_by_uid() -> dict[tuple[str, str], set[str]]:
    """(site, uid) -> the set of DISTINCT hand texts claiming it."""
    out: dict[tuple[str, str], set[str]] = {}
    for path in sorted(FIXTURES.glob("*.txt")):
        gg = path.name.startswith("gg_")
        site = "GGPoker" if gg else "PokerStars"
        text = path.read_text()
        for chunk in (split_ggpoker(text) if gg else split_pokerstars(text)):
            head = chunk.lstrip().split("\n", 1)[0]
            if "Hand #" not in head:
                continue                      # a deliberately broken header
            uid = head.split("Hand #", 1)[1].split(":", 1)[0].strip()
            out.setdefault((site, uid), set()).add(chunk.strip())
    return out


def test_no_two_different_hands_share_a_hand_number():
    collisions = {k: v for k, v in _hands_by_uid().items() if len(v) > 1}
    assert not collisions, (
        "these (site, hand_uid) keys are claimed by more than one distinct "
        "hand, so importing both in one session silently drops one: "
        + ", ".join(f"{s} #{u} ({len(v)} hands)" for (s, u), v in collisions.items())
    )
