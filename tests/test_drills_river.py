"""River drills — the first postflop tier (tier 2, solver-graded).

The honesty contract mirrors the chart artifacts: the checked-in strategies
are only trusted because the FAST suite re-certifies every board's Nash gap
from scratch (`SubgameSolver.exploitability(avg=stored)` on a freshly built
tree), and every drill's range_ctx names the game actually solved — fixture
preflop ranges, checked-down line, 33/75/jam grid, measured gap.
"""

import pytest

from pokerlab.drills.river import (BOARDS_PER_TEXTURE, PRETTY, TARGET_GAP_BB,
                                   load_river_solves, river_boards,
                                   river_drills)
from pokerlab.types import TIER_SOLVER


def test_boards_cover_every_texture_with_full_runouts():
    boards = river_boards()
    textures = {t for _, t in boards}
    from pokerlab.solver.texture import TEXTURES
    assert textures == set(TEXTURES)
    for b, _ in boards:
        assert len(b) == 10                       # five concrete cards
        cards = {b[i:i + 2] for i in range(0, 10, 2)}
        assert len(cards) == 5                    # no duplicate card
    per_tex = [sum(1 for _, t2 in boards if t2 == t) for t in textures]
    assert all(n <= BOARDS_PER_TEXTURE for n in per_tex)


def test_every_stored_board_recertifies_from_scratch():
    """No black boxes: rebuild each tree deterministically, inject the stored
    average strategies, and MEASURE the gap the artifact claims."""
    solves = load_river_solves()
    assert len(solves) == len(river_boards())
    for board, (solver, avg, stored_gap) in solves.items():
        assert stored_gap <= TARGET_GAP_BB, (board, stored_gap)
        measured = solver.exploitability(avg=avg)
        # float32 storage + platform drift headroom, not behavior ([R4-3])
        assert measured <= TARGET_GAP_BB * 1.5, (board, measured)


def test_river_drill_shape_and_honest_grading():
    drills = river_drills()
    assert drills, "no river drills generated"
    d = drills[0]
    assert d.kind == "river"
    assert d.tier == TIER_SOLVER
    assert len(d.board) == 5
    assert len(d.hero_cards) == 2
    # hero's concrete cards are consistent with the class label drilled
    assert d.hand_label[0] in d.hero_cards[0] + d.hero_cards[1]
    # actions are the solved grid, priced — every legal action has a real EV
    assert set(d.legal_actions) <= set(PRETTY.values())
    for a in d.legal_actions:
        ev, freq = d.solution.actions[a]
        assert isinstance(ev, float) and isinstance(freq, float)
    # nothing off-tree is offered: the grid is the complete solved action set
    assert d.off_tree_actions == ()
    # provenance discloses the game: fixture ranges, line, grid, measured gap
    for token in ("flops25", "unfiltered", "checked", "measured_gap"):
        assert token in d.solution.range_ctx, token
    assert d.solution.source == "subgame_solver"
    assert d.table == ("BB", "BTN")
    assert d.position == "BB"


def test_hero_combo_never_collides_with_the_board():
    for d in river_drills():
        assert not set(d.hero_cards) & set(d.board), d.drill_id


def test_drill_ids_unique_across_boards():
    ids = [d.drill_id for d in river_drills()]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("cls", ["72o"])
def test_out_of_range_classes_get_no_drill(cls):
    """The BB fixture range doesn't defend 72o preflop — grading a hand the
    range never holds would be grading against a zero-reach fabrication."""
    assert not [d for d in river_drills() if d.hand_label == cls]
