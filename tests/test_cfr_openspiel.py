"""Slice B — REAL cross-checks of the exploitability instrument vs OpenSpiel.

Chip-unit mapping (documented): our games replicate OpenSpiel's payoffs exactly
(proven by a full lockstep replay of all terminals), and our `nash_conv` returns
the SUM over players of best-response gain, whereas `pyspiel.exploitability`
returns that sum divided by num_players. Hence for 2-player zero-sum:

    pyspiel.exploitability(game, policy)  ==  our nash_conv(game, policy) / 2

We build OpenSpiel's policy mapping from OUR strategy by replaying each info
state's history into our game (the two infoset partitions are identical, and
our action integers match OpenSpiel's), so this compares the very same strategy
in both engines.
"""

import pytest

pyspiel = pytest.importorskip("pyspiel")

from pokerlab.cfr.cfr import CFRSolver
from pokerlab.cfr.exploit import nash_conv
from pokerlab.cfr.game import build_tree, uniform_profile
from pokerlab.cfr.kuhn import KuhnPoker
from pokerlab.cfr.leduc import LeducPoker

pytestmark = pytest.mark.openspiel

_GAMES = {"kuhn_poker": KuhnPoker, "leduc_poker": LeducPoker}


def _to_openspiel_mapping(os_game, game, profile):
    """OUR profile -> {info_state_string: [(action, prob), ...]} via history replay."""
    root = game.initial_states()[0][0]
    mapping = {}

    def walk(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for a, _ in state.chance_outcomes():
                walk(state.child(a))
            return
        iss = state.information_state_string()
        if iss not in mapping:
            mine = root
            for a in state.history():
                mine = game.apply(mine, a)
            dist = profile[game.infoset_key(mine, state.current_player())]
            mapping[iss] = [(a, dist[a]) for a in state.legal_actions()]
        for a in state.legal_actions():
            walk(state.child(a))

    walk(os_game.new_initial_state())
    return mapping


@pytest.mark.parametrize("name", list(_GAMES))
def test_terminal_payoffs_match_openspiel(name):
    """Lockstep replay: OUR returns equal OpenSpiel's on EVERY terminal history
    (proves identical chip units + game structure)."""
    game = _GAMES[name]()
    og = pyspiel.load_game(name)
    checked = 0

    def walk(state, mine):
        nonlocal checked
        if state.is_terminal():
            assert game.is_terminal(mine)
            assert game.returns(mine) == list(state.returns())
            checked += 1
            return
        if state.is_chance_node():
            for a, _ in state.chance_outcomes():
                walk(state.child(a), game.apply(mine, a))
            return
        assert sorted(game.legal_actions(mine)) == sorted(state.legal_actions())
        for a in state.legal_actions():
            walk(state.child(a), game.apply(mine, a))

    walk(og.new_initial_state(), game.initial_states()[0][0])
    assert checked == (30 if name == "kuhn_poker" else 5520)


@pytest.mark.parametrize("name", list(_GAMES))
def test_uniform_nashconv_matches_openspiel(name):
    game = _GAMES[name]()
    tree = build_tree(game)
    prof = uniform_profile(tree)
    og = pyspiel.load_game(name)
    mapping = _to_openspiel_mapping(og, game, prof)
    os_expl = pyspiel.exploitability(og, mapping)
    assert os_expl == pytest.approx(nash_conv(tree, prof) / 2.0, abs=1e-9)


@pytest.mark.parametrize("name,iters", [("kuhn_poker", 500), ("leduc_poker", 200)])
def test_cfr_plus_average_matches_openspiel(name, iters):
    game = _GAMES[name]()
    tree = build_tree(game)
    s = CFRSolver(tree, plus=True)
    s.run(iters)
    prof = s.average_profile()
    og = pyspiel.load_game(name)
    mapping = _to_openspiel_mapping(og, game, prof)
    os_expl = pyspiel.exploitability(og, mapping)
    # Same strategy, two independent engines -> must agree to numerical precision.
    assert os_expl == pytest.approx(nash_conv(tree, prof) / 2.0, abs=1e-6)
