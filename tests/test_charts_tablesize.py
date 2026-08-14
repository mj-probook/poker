"""Short-handed tables (2–9 players) for the ring and open games.

The chain solvers always took an arbitrary suffix table; what ships tonight
is the AXIS: named formations per table size, category tokens that keep
every existing 9-max/HU key byte-identical (the orphan rule, fourth
application), and artifacts covering the full size grid. Position names stay
the 9-max suffix names (6-max first-in is "LJ") — position relative to the
button is the strategic fact, and the range_ctx already states the table
size (`ring6`).
"""

import pytest

from pokerlab.charts.ring import RING_ORDER, solve_ring, table_for_size
from pokerlab.drills.categories import (open_category, resteal_category,
                                        ring_category)


def test_table_for_size_is_the_button_anchored_suffix():
    assert table_for_size(9) == RING_ORDER
    assert table_for_size(6) == ("LJ", "HJ", "CO", "BTN", "SB", "BB")
    assert table_for_size(3) == ("BTN", "SB", "BB")
    assert table_for_size(2) == ("SB", "BB")
    with pytest.raises(ValueError):
        table_for_size(1)
    with pytest.raises(ValueError):
        table_for_size(10)


def test_ring_category_gains_a_size_token_only_off_9max():
    # 9-max keys byte-identical — the orphan rule
    assert ring_category("CO", 10.0) == "COjam|preflop|jam|10"
    assert ring_category("SB", 10.0) == "SBjam.9max|preflop|jam|10"
    assert (ring_category("BB", 10.0, versus="CO")
            == "BBcall.vCO|preflop|call|10")
    # short-handed formations name their size — a 6-max first-in jam is a
    # different priced game (less dead money behind) than the 9-max one
    assert (ring_category("CO", 10.0, table_size=6)
            == "COjam.6max|preflop|jam|10")
    assert (ring_category("BB", 10.0, versus="CO", table_size=6)
            == "BBcall.vCO.6max|preflop|call|10")


def test_open_category_carries_the_size_from_the_formation():
    assert open_category("CO", 20.0) == "COopen|preflop|open|20"
    assert open_category("SBhu", 20.0) == "SBhuopen|preflop|open|20"
    assert open_category("CO.6max", 20.0) == "COopen.6max|preflop|open|20"
    assert (resteal_category("BB", "CO.6max", 2.2, 20.0)
            == "BBresteal.vCO.6max.r2.2|preflop|jam|20")


def test_short_table_ring_solve_certifies_and_orders_by_position():
    """Live solve at 3-max (tiny chain): the certificate holds, and the
    BTN first-in jams WIDER than the 9-max UTG at the same depth — fewer
    players left to wake up with a hand."""
    btn3 = solve_ring("BTN", 10.0, table=table_for_size(3))
    assert btn3.exploitability <= 1e-5
    from pokerlab.charts.ring import ring_solution
    utg9 = ring_solution("UTG", 10.0)
    assert btn3.jam_combos() > utg9.jam_combos()
    assert btn3.table == ("BTN", "SB", "BB")
