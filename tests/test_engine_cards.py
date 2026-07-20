"""Slice A: card codec round-trip + seeded deck determinism (impl doc §3)."""

from pokerlab.engine.cards import (
    Deck,
    card_from_str,
    card_rank,
    card_suit,
    card_to_str,
    full_deck,
    make_card,
)


def test_codec_round_trip_all_52() -> None:
    seen = set()
    for card in range(52):
        r, s = card_rank(card), card_suit(card)
        assert 2 <= r <= 14 and 0 <= s <= 3
        assert make_card(r, s) == card
        assert card_from_str(card_to_str(card)) == card
        seen.add(card_to_str(card))
    assert len(seen) == 52  # every card has a unique string


def test_known_card_strings() -> None:
    assert card_to_str(make_card(14, 3)) == "As"  # ace of spades = id 51
    assert make_card(14, 3) == 51
    assert card_from_str("2c") == 0
    assert card_from_str("aH") == card_from_str("Ah")  # case-insensitive


def test_make_card_validates() -> None:
    for bad in ((1, 0), (15, 0), (5, 4), (5, -1)):
        try:
            make_card(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")


def test_deck_seeded_determinism() -> None:
    assert Deck(42).deal(9) == Deck(42).deal(9)
    assert Deck(1).deal(9) != Deck(2).deal(9)


def test_deck_deals_unique_and_exhausts() -> None:
    d = Deck(7)
    dealt = d.deal(52)
    assert sorted(dealt) == full_deck()
    assert d.remaining == 0
    try:
        d.deal(1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError dealing past end")
