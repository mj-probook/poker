"""Hero decision extraction from a parsed hand (Slice F, impl doc §3).

Replays a `ParsedHand` through the Slice-A engine and snapshots the game state
at every point where the hero is to act. Each `Decision` carries exactly what
the tier router and the graders need: street, players-in-pot, effective stack,
pot, the hero's hole/board, the legal set, and the chosen action — plus the
routed tier and the raw `formation`/`street`/`action_type` fields the grader
turns into a canonical `leak_key` category (see `drills.categories`).
"""

from __future__ import annotations

from dataclasses import dataclass

import dataclasses

from pokerlab.engine.cards import card_rank, card_suit
from pokerlab.engine.state import Hand
from pokerlab.hh.model import ParsedHand
from pokerlab.hh.tiers import route_tier
from pokerlab.types import Action, GameState, STREETS, TournamentContext

_RANK_CHAR = {14: "A", 13: "K", 12: "Q", 11: "J", 10: "T",
              9: "9", 8: "8", 7: "7", 6: "6", 5: "5", 4: "4", 3: "3", 2: "2"}


def hand_label(hole: tuple[int, int]) -> str:
    """Two engine Card ints -> a 169-class label ('AA', 'AKs', 'AKo')."""
    c1, c2 = hole
    r1, r2 = card_rank(c1), card_rank(c2)
    hi, lo = max(r1, r2), min(r1, r2)
    if r1 == r2:
        return _RANK_CHAR[hi] + _RANK_CHAR[hi]
    suited = "s" if card_suit(c1) == card_suit(c2) else "o"
    return _RANK_CHAR[hi] + _RANK_CHAR[lo] + suited


def position_label(n: int, button: int, seat: int) -> str:
    """Positional label for a seat given table size and button (engine indices)."""
    if n == 2:
        return "SB" if seat == button else "BB"  # HU: button posts the SB
    sb, bb = (button + 1) % n, (button + 2) % n
    if seat == sb:
        return "SB"
    if seat == bb:
        return "BB"
    if seat == button:
        return "BTN"
    dist = (button - seat) % n  # 1 = right of button = CO, ...
    return {1: "CO", 2: "HJ", 3: "LJ", 4: "MP", 5: "UTG"}.get(dist, f"EP{dist}")


@dataclass
class Decision:
    index: int              # 0-based order within the hero's own decisions
    street: str
    seat: int               # hero engine index
    position: str
    num_in_pot: int         # non-folded players at decision time (2 == heads-up)
    pot: int                # chips in the pot before the action
    pot_bb: float
    eff_bb: float           # effective (smallest live) starting stack, in bb
    ante_bb: float          # per-player ante as a fraction of a big blind
    to_call: int
    opp_allin: bool         # a live opponent is already all-in (hero faces a jam)
    hole: tuple[int, int]
    board: tuple[int, ...]
    legal: list[Action]
    chosen: Action
    is_allin: bool          # the chosen action commits the hero's whole stack
    tier: int
    formation: str
    action_type: str
    # Frozen GameState snapshot at the decision (tournament context attached),
    # so downstream (tier-2 solver / SpotKey) is self-contained. Optional so a
    # Decision can still be hand-built in tests without a full engine state.
    game_state: GameState | None = None
    # The persisted leak_key is the canonical category (drills.categories),
    # assigned by the grader — jam/fold spots gain a depth bucket, so it is not
    # a pure function of these raw fields (see hh.grade).


def _action_type(chosen: Action, is_allin: bool) -> str:
    label, _ = chosen
    if label == "fold":
        return "fold"
    if is_allin and label in ("bet", "raise", "allin"):
        return "jam"
    return label


def extract_decisions(parsed: ParsedHand) -> list[Decision]:
    hand = Hand(parsed.setup)
    n = hand.n
    bb = parsed.setup.bb
    hero = parsed.hero
    tc = TournamentContext(
        payouts=(0,),
        players_remaining=n,
        stacks_all=tuple(parsed.setup.stacks),
        bb=bb,
        ante=parsed.setup.ante,
    )
    out: list[Decision] = []
    k = 0
    for seat, action in parsed.actions:
        if hand.to_act == seat == hero:
            street = STREETS[hand.street_idx]
            num_in_pot = sum(not f for f in hand.folded)
            max_to = hand.street_bet[hero] + hand.stack_left[hero]
            label, amount = action
            is_allin = label in ("bet", "raise", "allin") and amount >= max_to
            if label == "call" and hand.current_bet >= max_to:
                is_allin = True  # calling commits the hero's whole stack
            eff_bb = min(
                parsed.setup.stacks[i] for i in range(n) if not hand.folded[i]
            ) / bb
            atype = _action_type(action, is_allin)
            out.append(Decision(
                index=k,
                street=street,
                seat=hero,
                position=position_label(n, hand.button, hero),
                num_in_pot=num_in_pot,
                pot=hand.pot,
                pot_bb=hand.pot / bb,
                eff_bb=eff_bb,
                ante_bb=parsed.setup.ante / bb,
                to_call=hand.current_bet - hand.street_bet[hero],
                opp_allin=any(
                    hand.allin[i] and not hand.folded[i]
                    for i in range(n) if i != hero
                ),
                hole=parsed.setup.hole[hero],  # type: ignore[arg-type]
                board=tuple(hand._visible_board()),
                legal=hand.legal_actions(),
                chosen=action,
                is_allin=is_allin,
                tier=route_tier(street, num_in_pot),
                formation=f"{n}max:{position_label(n, hand.button, hero)}",
                action_type=atype,
                game_state=dataclasses.replace(hand.game_state, tournament=tc),
            ))
            k += 1
        hand.apply(action)
    return out
