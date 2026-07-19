"""``resolve_spot_key`` — GameState → SpotKey (plan §1; impl doc §1, §3 Slice D).

The shared library/grading lookup key. A SpotKey is
``(formation, stack_bucket, board_bucket)``:

  * **formation** — the preflop position/action configuration, e.g.
    ``"BTNopen_BBcall"``. Derived by segmenting the preflop betting round out of
    the flat action history (blinds-aware) and naming each voluntary aggressive
    or calling action by position.
  * **stack_bucket** — effective stack in bb snapped to the nearest supported
    depth in {10, 20, 40, 100}.
  * **board_bucket** — ``f"{iso_class}:{texture}"`` (solver/iso.py + texture.py).

Effective stack and blinds are read from ``state.tournament`` (the working unit
is big blinds); a state without tournament context cannot be keyed.
"""

from __future__ import annotations

from pokerlab.engine.tournament import effective_bb
from pokerlab.solver import iso, texture
from pokerlab.types import Action, GameState, SpotKey

_STACK_BUCKETS: tuple[int, ...] = (10, 20, 40, 100)
_RAISE_VERBS = ("open", "3bet", "4bet", "5bet")


def _position_name(seat: int, seats: int, button: int) -> str:
    """Table-position label. BTN/SB/BB/CO are exact; intermediate names are
    conventional approximations (not load-bearing for lookups)."""
    off = (seat - button) % seats
    if seats == 2:
        return "BTN" if off == 0 else "BB"
    if off == 0:
        return "BTN"
    if off == 1:
        return "SB"
    if off == 2:
        return "BB"
    if off == seats - 1:
        return "CO"
    early = ("UTG", "UTG1", "UTG2", "MP", "MP1", "HJ")
    return early[off - 3] if off - 3 < len(early) else f"P{off}"


def _seat_roles(seats: int, button: int) -> tuple[int, int]:
    """(sb_seat, bb_seat) under the engine convention."""
    if seats == 2:
        return button, (button + 1) % seats
    return (button + 1) % seats, (button + 2) % seats


def _preflop_actions(state: GameState, bb: int, sb: int) -> list[tuple[int, Action]]:
    """The prefix of the action history that belongs to the preflop round.

    Replays only the betting arithmetic (no cards needed): amounts are
    street "raise-to" commitments, so the round closes once every non-folded
    seat has acted and matched the current bet (BB option included).
    """
    sb_seat, bb_seat = _seat_roles(state.seats, button=state.button)
    committed = {sb_seat: sb, bb_seat: bb}
    current_bet = bb
    folded: set[int] = set()
    acted: set[int] = set()
    out: list[tuple[int, Action]] = []
    for seat, action in state.action_history:
        label, amount = action
        if label == "fold":
            folded.add(seat)
            acted.add(seat)
        elif label == "check":
            acted.add(seat)
        elif label == "call":
            committed[seat] = current_bet
            acted.add(seat)
        else:  # bet / raise / allin
            committed[seat] = amount
            if amount > current_bet:
                current_bet = amount
                acted = {seat}
            else:
                acted.add(seat)
        out.append((seat, action))
        live = [s for s in range(state.seats) if s not in folded]
        if all(s in acted and committed.get(s, 0) == current_bet for s in live):
            break  # preflop round closed
    return out


def _formation(state: GameState) -> str:
    """Compact preflop formation string, e.g. ``BTNopen_BBcall``."""
    pre = _preflop_actions(state, bb=state.tournament.bb, sb=state.tournament.bb // 2)
    _, bb_seat = _seat_roles(state.seats, state.button)
    current_bet = state.tournament.bb
    raises = 0
    parts: list[str] = []
    for seat, (label, amount) in pre:
        pos = _position_name(seat, state.seats, state.button)
        if label in ("bet", "raise", "allin") and amount > current_bet:
            parts.append(f"{pos}{_RAISE_VERBS[min(raises, len(_RAISE_VERBS) - 1)]}")
            raises += 1
            current_bet = amount
        elif label == "call":
            parts.append(f"{pos}call")
        # folds / checks (incl. BB check in a limped pot) carry no formation token
    return "_".join(parts) if parts else "limped"


def _stack_bucket(state: GameState) -> int:
    """Effective stack in bb snapped to the nearest supported depth."""
    folded = {seat for seat, (label, _) in state.action_history if label == "fold"}
    live = [i for i in range(state.seats) if i not in folded]
    eff_chips = min(state.stacks[i] for i in live)
    eff_bb = effective_bb(eff_chips, state.tournament)
    return min(_STACK_BUCKETS, key=lambda b: (abs(b - eff_bb), b))


def resolve_spot_key(state: GameState) -> SpotKey:
    """Resolve a GameState to its library/grading SpotKey (plan §1)."""
    if state.tournament is None:
        raise ValueError("resolve_spot_key needs state.tournament (bb, stacks) set")
    flop = tuple(state.board[:3])
    if len(flop) < 3:
        raise ValueError("board_bucket needs at least a 3-card flop")
    board_bucket = f"{iso.iso_class(flop)}:{texture.texture(flop)}"
    return SpotKey(
        formation=_formation(state),
        stack_bucket=_stack_bucket(state),
        board_bucket=board_bucket,
    )
