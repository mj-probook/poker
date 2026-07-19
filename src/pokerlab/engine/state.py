"""NLHE single-hand state machine (impl doc §3 Slice A; plan §3b).

`Hand` is the engine's mutable state machine. `types.GameState` is the frozen
data contract; `Hand.game_state` snapshots into it (revealed board, start
stacks, pot, street, action history, to-act). Positions/action-order/side-pot/
odd-chip rules are validated against PokerKit by the differential suite.

Seat/position convention (matches PokerKit so the differential compares seat
for seat): for >=3 players seat ``(button+1)%n`` is the SB and ``(button+2)%n``
the BB; heads-up the button posts the SB. Amounts in an `Action` are the
resulting *street commitment* ("raise-to"), 0 for fold/check.
"""

from __future__ import annotations

from dataclasses import dataclass

from pokerlab.engine.cards import card_to_str
from pokerlab.engine.evaluator import rank_showdown
from pokerlab.types import Action, Card, GameState, STREETS, TournamentContext

# cards visible by street index 0=preflop..3=river (4 = showdown/terminal)
_BOARD_BY_STREET = (0, 3, 4, 5, 5)


@dataclass
class HandSetup:
    """Everything needed to start (and replay) a hand."""

    stacks: tuple[int, ...]
    button: int
    bb: int
    sb: int
    ante: int
    hole: tuple[tuple[Card, Card] | None, ...]
    board: tuple[Card, ...]  # up to 5 cards; revealed progressively
    # Big-blind ante: ONE ante posted by the big blind for the whole table
    # (the modern tournament format). Dead money — it never becomes a live bet,
    # so it does not raise the amount anyone has to call. Mutually exclusive
    # with `ante` in practice; both paths are independent (round-1 finding [12]).
    bb_ante: int = 0


class Hand:
    def __init__(self, setup: HandSetup) -> None:
        self.setup = setup
        n = len(setup.stacks)
        self.n = n
        self.button = setup.button
        self.start_stacks = list(setup.stacks)
        self.bb = setup.bb
        self.sb = setup.sb
        self.ante = setup.ante
        self.bb_ante = setup.bb_ante
        self.hole = list(setup.hole)
        self.full_board = list(setup.board)

        self.contrib = [0] * n
        self.street_bet = [0] * n
        self.stack_left = list(setup.stacks)
        self.folded = [False] * n
        self.allin = [False] * n
        self.action_history: list[tuple[int, Action]] = []
        self.street_idx = 0
        self._terminal = False
        self.to_act: int | None = None
        self._queue: list[int] = []

        # per-street betting bookkeeping (see class docstring / apply())
        self.min_raise_inc = self.bb  # min legal raise increment (last full raise)
        self.acted: set[int] = set()  # seats acted since last full raise
        self.consec_short: list[int] = []  # consecutive all-in short-raise incs

        self._post_antes_and_blinds()
        self.current_bet = max(self.street_bet) if any(self.street_bet) else 0
        self._build_queue(self._first_preflop())
        self._maybe_close_if_no_action()

    # ---- positions -------------------------------------------------------
    def _sb_seat(self) -> int:
        return self.button if self.n == 2 else (self.button + 1) % self.n

    def _bb_seat(self) -> int:
        return (self.button + 1) % self.n if self.n == 2 else (self.button + 2) % self.n

    def _first_preflop(self) -> int:
        return self.button if self.n == 2 else (self.button + 3) % self.n

    def _first_postflop(self) -> int:
        return (self.button + 1) % self.n

    # ---- setup -----------------------------------------------------------
    def _commit(self, seat: int, pay: int) -> None:
        pay = min(pay, self.stack_left[seat])
        self.street_bet[seat] += pay
        self.contrib[seat] += pay
        self.stack_left[seat] -= pay
        if self.stack_left[seat] == 0:
            self.allin[seat] = True

    def _post_ante(self, seat: int, amount: int) -> None:
        """Charge a dead-money ante: contribution only, never a live bet."""
        amt = min(amount, self.stack_left[seat])
        self.contrib[seat] += amt
        self.stack_left[seat] -= amt
        if self.stack_left[seat] == 0:
            self.allin[seat] = True

    def _post_antes_and_blinds(self) -> None:
        if self.ante:
            for i in range(self.n):
                self._post_ante(i, self.ante)
        if self.bb_ante:
            # one ante for the table, paid by the big blind, posted before the
            # blinds (it is dead money, not part of the BB's live blind)
            self._post_ante(self._bb_seat(), self.bb_ante)
        self._commit(self._sb_seat(), self.sb)
        self._commit(self._bb_seat(), self.bb)

    def _order_from(self, start: int) -> list[int]:
        return [(start + k) % self.n for k in range(self.n)]

    def _active(self, seat: int) -> bool:
        return not self.folded[seat] and not self.allin[seat]

    def _build_queue(self, start: int) -> None:
        self._queue = [i for i in self._order_from(start) if self._active(i)]
        self.to_act = self._queue[0] if self._queue else None

    def _maybe_close_if_no_action(self) -> None:
        # If <=1 player can act (all others all-in/folded), skip betting.
        if self.to_act is None or len(self._queue) == 0:
            self._advance_street()

    # ---- action space ----------------------------------------------------
    def _to_call(self, seat: int) -> int:
        return self.current_bet - self.street_bet[seat]

    def _can_raise(self, seat: int) -> bool:
        """Mirror PokerKit's raise-permission rules."""
        max_to = self.street_bet[seat] + self.stack_left[seat]
        if max_to <= self.current_bet:
            return False  # covered: can only call/fold (all-in for less)
        # a non-full all-in raise does not reopen raising for players who acted
        if (
            self.consec_short
            and sum(self.consec_short) < self.min_raise_inc
            and seat in self.acted
        ):
            return False
        # need someone else who can still put in more than the current bet
        if not any(
            i != seat
            and not self.folded[i]
            and self.street_bet[i] + self.stack_left[i] > self.current_bet
            for i in range(self.n)
        ):
            return False
        return True

    def raise_bounds(self, seat: int | None = None) -> tuple[int, int] | None:
        """(min_to, max_to) legal raise-to amounts, or None if no raise legal.

        If a full raise is impossible but the player has chips beyond a call,
        the only legal raise is an all-in for less: (max_to, max_to).
        """
        seat = self.to_act if seat is None else seat
        if seat is None or not self._can_raise(seat):
            return None
        max_to = self.street_bet[seat] + self.stack_left[seat]
        min_full = self.current_bet + self.min_raise_inc
        if max_to < min_full:
            return (max_to, max_to)  # short all-in raise only
        return (min_full, max_to)

    def legal_actions(self) -> list[Action]:
        seat = self.to_act
        if seat is None:
            return []
        acts: list[Action] = []
        to_call = self._to_call(seat)
        max_to = self.street_bet[seat] + self.stack_left[seat]
        if to_call == 0:
            acts.append(("check", 0))
        else:
            acts.append(("fold", 0))
            call_to = min(self.current_bet, max_to)
            label = "allin" if call_to == max_to else "call"
            acts.append((label, call_to))
        bounds = self.raise_bounds(seat)
        if bounds is not None:
            min_to, top = bounds
            if min_to < top:
                label = "raise" if self.current_bet > 0 else "bet"
                acts.append((label, min_to))
            acts.append(("allin", top))
        return acts

    # ---- applying actions ------------------------------------------------
    def apply(self, action: Action) -> None:
        seat = self.to_act
        if seat is None or self._terminal:
            raise ValueError("no player to act")
        label, amount = action
        max_to = self.street_bet[seat] + self.stack_left[seat]
        to_call = self._to_call(seat)

        if label == "fold":
            self.folded[seat] = True
            self._after_passive(seat, ("fold", 0))
        elif label == "check":
            if to_call != 0:
                raise ValueError("cannot check facing a bet")
            self._after_passive(seat, ("check", 0))
        elif label in ("call", "bet", "raise", "allin"):
            target = min(self.current_bet, max_to) if label == "call" else amount
            if target < self.street_bet[seat]:
                raise ValueError("amount below current commitment")
            if target > max_to:
                raise ValueError("amount exceeds stack")
            if target > self.current_bet:
                self._apply_raise(seat, label, target)
            else:  # a call (possibly all-in for less)
                self._commit(seat, target - self.street_bet[seat])
                self._after_passive(seat, (label, target))
        else:
            raise ValueError(f"unknown action label: {label}")

    def _apply_raise(self, seat: int, label: str, target: int) -> None:
        bounds = self.raise_bounds(seat)
        if bounds is None:
            raise ValueError("raise not permitted here")
        min_to, top = bounds
        if not (target == top or min_to <= target <= top):
            raise ValueError(f"illegal raise-to {target}, bounds {bounds}")
        prev_bet = self.current_bet
        inc = target - prev_bet
        self._commit(seat, target - self.street_bet[seat])
        self.current_bet = target
        # acted set: a full raise resets it; a short all-in raise appends.
        if inc >= self.min_raise_inc:
            self.acted = {seat}
        else:
            self.acted.add(seat)
        self.min_raise_inc = max(self.min_raise_inc, inc)
        if self.stack_left[seat] == 0:
            self.consec_short.append(inc)
        else:
            self.consec_short.clear()
        # everyone else still active must respond to the raise
        self._queue = [
            i for i in self._order_from((seat + 1) % self.n) if self._active(i) and i != seat
        ]
        self.action_history.append((seat, (label, target)))
        self._finalize_turn()

    def _after_passive(self, seat: int, recorded: Action) -> None:
        self.acted.add(seat)
        self.action_history.append((seat, recorded))
        if self._queue and self._queue[0] == seat:
            self._queue.pop(0)
        else:
            self._queue = [i for i in self._queue if i != seat]
        self._finalize_turn()

    def _finalize_turn(self) -> None:
        if sum(not f for f in self.folded) <= 1:  # everyone else folded
            self._finish()
            return
        self.to_act = self._queue[0] if self._queue else None
        if self.to_act is None:
            self._advance_street()

    # ---- street transitions ---------------------------------------------
    def _advance_street(self) -> None:
        self.street_bet = [0] * self.n
        self.current_bet = 0
        self.min_raise_inc = self.bb
        self.acted = set()
        self.consec_short = []
        alive = [i for i in range(self.n) if not self.folded[i]]
        if len(alive) <= 1:
            self._finish()
            return
        can_act = [i for i in alive if not self.allin[i]]
        if len(can_act) <= 1:
            self.street_idx = 4  # run out board to showdown
            self._finish()
            return
        if self.street_idx >= 3:
            self.street_idx = 4
            self._finish()
            return
        self.street_idx += 1
        self._build_queue(self._first_postflop())

    def _finish(self) -> None:
        self._terminal = True
        self.to_act = None
        self._queue = []

    # ---- terminal / results ---------------------------------------------
    def is_terminal(self) -> bool:
        return self._terminal

    @property
    def pot(self) -> int:
        return sum(self.contrib)

    def _visible_board(self) -> list[Card]:
        return self.full_board[: _BOARD_BY_STREET[self.street_idx]]

    def payoffs(self) -> list[int]:
        """Chip deltas per seat (winnings - contributed). Zero-sum."""
        if not self._terminal:
            raise ValueError("hand not terminal")
        winnings = self._distribute()
        return [winnings[i] - self.contrib[i] for i in range(self.n)]

    def final_stacks(self) -> list[int]:
        winnings = self._distribute()
        return [self.start_stacks[i] - self.contrib[i] + winnings[i] for i in range(self.n)]

    def _distribute(self) -> list[int]:
        """Award main + side pots, replicating PokerKit's pot formation exactly.

        Pots are cut at every distinct contribution level, adjacent pots with
        an identical eligible list are merged, each pot's best hand(s) split it,
        and the odd chip goes to the lowest-index winner. Uncalled bets fall out
        as a top pot with a single eligible player.

        The eligible list depends on how the hand ended (matching PokerKit's
        showdown/hand-killing behaviour):

        * If every non-folded player is all-in, PokerKit's all-in showdown
          turns *all* hands face-up and kills nothing, so eligibility at each
          level is simply the non-folded contributors. Thin same-winner layers
          separated by a losing all-in stay distinct (each may mint an odd
          chip).
        * Otherwise (a player still has chips at showdown) PokerKit auto-mucks
          losing hands, so eligibility is the *survivors* — players who hold the
          best hand among everyone who committed at least as much as they did.
          Losing all-ins drop out, merging their layers.
        """
        n = self.n
        winnings = [0] * n
        contrib = self.contrib
        nonfolded = [i for i in range(n) if not self.folded[i]]

        if len(nonfolded) <= 1:  # uncontested: sole player reclaims every layer
            keep = set(nonfolded)
            ranks: dict[int, int] = {}
        else:
            board = self.full_board[:5]
            ranks = {i: rank_showdown(self.hole[i], board) for i in nonfolded}  # type: ignore[arg-type]
            if all(self.allin[i] for i in nonfolded):
                keep = set(nonfolded)
            else:  # auto-muck losers -> keep survivors
                keep = {
                    i
                    for i in nonfolded
                    if ranks[i] == min(ranks[j] for j in nonfolded if contrib[j] >= contrib[i])
                }

        levels = sorted({c for c in contrib if c > 0})
        prev = 0
        pots: list[list] = []  # [amount, eligible_player_indices]
        for level in levels:
            amount = (level - prev) * sum(1 for i in range(n) if contrib[i] >= level)
            prev = level
            pidx = tuple(i for i in range(n) if contrib[i] >= level and i in keep)
            while pots and pots[-1][1] == pidx:  # merge same-eligibility layers
                amount += pots.pop()[0]
            pots.append([amount, pidx])

        for amount, pidx in pots:
            if not pidx:  # only folded/killed players staked this layer
                continue
            if len(pidx) == 1:
                winnings[pidx[0]] += amount  # incl. returned uncalled bet
                continue
            best = min(ranks[i] for i in pidx)
            winners = [i for i in pidx if ranks[i] == best]  # ascending index
            q, r = divmod(amount, len(winners))
            for k, w in enumerate(winners):
                winnings[w] += q + (r if k == 0 else 0)  # odd chip -> lowest index
        return winnings

    # ---- contract snapshot ----------------------------------------------
    @property
    def game_state(self) -> GameState:
        return GameState(
            seats=self.n,
            stacks=list(self.start_stacks),
            button=self.button,
            board=self._visible_board(),
            hole=list(self.hole),
            pot=self.pot,
            street=STREETS[self.street_idx],
            action_history=list(self.action_history),
            to_act=self.to_act,
            tournament=None,
        )

    # ---- replay ----------------------------------------------------------
    @classmethod
    def replay(cls, setup: HandSetup, actions: list[tuple[int, Action]]) -> "Hand":
        """Rebuild terminal state by replaying a (seat, action) list."""
        hand = cls(setup)
        for seat, action in actions:
            if hand.to_act != seat:
                raise ValueError(f"replay: expected seat {hand.to_act}, got {seat}")
            hand.apply(action)
        return hand

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        board = "".join(card_to_str(c) for c in self._visible_board())
        return (
            f"Hand(street={STREETS[self.street_idx]} to_act={self.to_act} "
            f"pot={self.pot} board={board or '-'} contrib={self.contrib})"
        )


def new_hand(
    stacks: list[int] | tuple[int, ...],
    button: int,
    *,
    bb: int,
    sb: int | None = None,
    ante: int = 0,
    hole: list[tuple[Card, Card] | None] | tuple[tuple[Card, Card] | None, ...],
    board: list[Card] | tuple[Card, ...] = (),
) -> Hand:
    """Convenience constructor. Defaults SB = bb // 2."""
    return Hand(
        HandSetup(
            stacks=tuple(stacks),
            button=button,
            bb=bb,
            sb=bb // 2 if sb is None else sb,
            ante=ante,
            hole=tuple(hole),
            board=tuple(board),
        )
    )
