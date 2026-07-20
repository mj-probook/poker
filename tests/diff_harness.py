"""Differential harness: engine vs PokerKit (impl doc §3 Slice A, M0 exit).

Generates seeded random NLHE hands, plays them through our engine with a random
legal-action policy, then reconstructs the *identical* hand in PokerKit and
compares final per-seat stacks. PokerKit is the presumed-correct oracle.

Reusable by Slice H (M6): the vec-env's ``SingleEnvAdapter`` runs the same
generator + comparison to prove the Rust/numpy port matches the Python engine.

Seat layout is chosen to match PokerKit's fixed convention (n>=3: seat0=SB,
seat1=BB, button=n-1; HU: seat1=SB/button) so stacks compare seat-for-seat.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from pokerkit import Automation, Mode, NoLimitTexasHoldem

from pokerlab.engine.cards import Deck, cards_to_str
from pokerlab.engine.evaluator import rank_showdown
from pokerlab.engine.state import Hand, HandSetup
from pokerlab.types import Action

BB = 100
SB = 50

_AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.CARD_BURNING,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
    Automation.RUNOUT_COUNT_SELECTION,
)


def random_setup(seed: int) -> HandSetup:
    """A seeded random hand. Stacks >= 2bb so blinds/antes always post fully
    (no ante-trimming edge); arbitrary chip counts so odd-chip splits arise."""
    rng = random.Random(seed)
    n = rng.randint(2, 9)
    ante = rng.choice([0, 0, 0, 10, 25])
    stacks = []
    for _ in range(n):
        lo, hi = (2 * BB, 25 * BB) if rng.random() < 0.45 else (25 * BB, 150 * BB)
        stacks.append(rng.randint(lo, hi))
    button = 1 if n == 2 else n - 1
    deck = Deck((seed * 2654435761) & 0xFFFFFFFF)
    cards = deck.deal(2 * n + 5)
    hole = tuple((cards[2 * i], cards[2 * i + 1]) for i in range(n))
    board = tuple(cards[2 * n : 2 * n + 5])
    return HandSetup(stacks=tuple(stacks), button=button, bb=BB, sb=SB,
                     ante=ante, hole=hole, board=board)


def short_stack_setup(seed: int) -> HandSetup:
    """A seeded random hand that DELIBERATELY includes sub-blind stacks.

    `random_setup` pins every stack at >= 2bb to dodge the ante/blind-trimming
    edge — which also means it never generates the states where a blind or ante
    puts a player all-in before anyone acts. That blind spot hid wave-2 [E1]:
    with stacks >= 2bb the engine and PokerKit never disagreed (0/4000), and
    with sub-blind stacks they disagreed on 312/4000 (7.8%).

    Those states are not exotic — a player short of a blind is routine in the
    MTT hands this project exists to grade, so they belong in the oracle.
    """
    rng = random.Random(seed)
    n = rng.randint(2, 6)
    ante = rng.choice([0, 0, 10, 25])
    stacks = []
    for _ in range(n):
        if rng.random() < 0.45:            # short: at or below one big blind
            stacks.append(rng.randint(5, 2 * BB))
        else:
            stacks.append(rng.randint(2 * BB, 60 * BB))
    button = 1 if n == 2 else n - 1
    deck = Deck((seed * 2654435761) & 0xFFFFFFFF)
    cards = deck.deal(2 * n + 5)
    hole = tuple((cards[2 * i], cards[2 * i + 1]) for i in range(n))
    board = tuple(cards[2 * n : 2 * n + 5])
    return HandSetup(stacks=tuple(stacks), button=button, bb=BB, sb=SB,
                     ante=ante, hole=hole, board=board)


def bb_ante_setup(seed: int) -> HandSetup:
    """A seeded random hand with a BIG-BLIND ANTE — one player posts for the table.

    The third permanent differential axis (wave-3 [A2]), added for the same
    reason as `short_stack_setup`: the existing generators emit only uniform
    antes (`ante`, every seat pays) or none, so no hand they can produce has a
    single-payer dead-money contribution. That is precisely the shape [A1] lived
    in — the engine returned a lone ante to a losing BB as though it were an
    uncalled bet — and both existing axes were green through all of it.

    BB ante is the standard modern MTT structure, so this is the common case,
    not an exotic one.
    """
    rng = random.Random(seed ^ 0xBBA07E)
    n = rng.randint(2, 6)
    bb_ante = rng.choice([BB, BB, BB // 2, 2 * BB])
    stacks = []
    for _ in range(n):
        if rng.random() < 0.35:        # short enough to be all-in from the ante
            stacks.append(rng.randint(5, 3 * BB))
        else:
            stacks.append(rng.randint(3 * BB, 60 * BB))
    button = 1 if n == 2 else n - 1
    deck = Deck((seed * 2654435761) & 0xFFFFFFFF)
    cards = deck.deal(2 * n + 5)
    hole = tuple((cards[2 * i], cards[2 * i + 1]) for i in range(n))
    board = tuple(cards[2 * n : 2 * n + 5])
    return HandSetup(stacks=tuple(stacks), button=button, bb=BB, sb=SB,
                     ante=0, hole=hole, board=board, bb_ante=bb_ante)


def random_action(hand: Hand, rng: random.Random) -> Action:
    """Pick a random *legal* action, biased toward reaching showdowns while
    still frequently raising/jamming for side-pot coverage."""
    acts = hand.legal_actions()
    labels = {a[0]: a for a in acts}
    roll = rng.random()
    if "fold" in labels and roll < 0.12:
        return ("fold", 0)
    bounds = hand.raise_bounds()
    if bounds is not None and roll < 0.55:
        min_to, max_to = bounds
        if min_to == max_to or roll < 0.22:
            return ("allin", max_to)
        to = rng.randint(min_to, max_to)
        label = "allin" if to == max_to else ("raise" if hand.current_bet > 0 else "bet")
        return (label, to)
    if "check" in labels:
        return ("check", 0)
    if "call" in labels:
        return labels["call"]
    if "allin" in labels:  # all-in call for less
        return labels["allin"]
    return acts[0]


def play_engine(setup: HandSetup, seed: int, action_fn=random_action) -> Hand:
    """Play a full hand through our engine. ``action_fn(hand, rng) -> Action``
    defaults to the random legal policy; Slice H passes a discrete-encoding
    policy to run the identical differential through the vectorized-env path."""
    rng = random.Random(seed ^ 0x5DEECE66D)
    hand = Hand(setup)
    while not hand.is_terminal():
        hand.apply(action_fn(hand, rng))
    return hand


def pokerkit_final_stacks(setup: HandSetup, action_history: list[tuple[int, Action]]) -> list[int]:
    """Replay the same hand in PokerKit; return final per-seat stacks."""
    n = len(setup.stacks)
    blinds = (setup.sb, setup.bb) + (0,) * (n - 2)
    s = NoLimitTexasHoldem.create_state(
        _AUTOMATIONS, not setup.bb_ante, _pk_antes(setup), blinds, setup.bb,
        tuple(setup.stacks), n, mode=Mode.TOURNAMENT,
    )
    for i in range(n):
        s.deal_hole(cards_to_str(setup.hole[i]))

    board = setup.board
    dealt = 0

    def drain_board() -> None:
        nonlocal dealt
        while s.can_deal_board():
            cnt = s.board_dealing_count
            s.deal_board(cards_to_str(board[dealt : dealt + cnt]))
            dealt += cnt

    drain_board()
    for seat, (label, amount) in action_history:
        if s.actor_index != seat:
            raise AssertionError(
                f"action-order mismatch: pokerkit actor {s.actor_index}, engine seat {seat}"
            )
        if label == "fold":
            s.fold()
        elif label in ("check", "call"):
            s.check_or_call()
        elif label in ("bet", "raise"):
            s.complete_bet_or_raise_to(amount)
        elif label == "allin":
            if amount > max(s.bets):  # all-in raise (full or short)
                s.complete_bet_or_raise_to(amount)
            else:  # all-in call for less
                s.check_or_call()
        else:  # pragma: no cover
            raise ValueError(f"unknown label {label}")
        drain_board()
    _drain_forced_checks(s, drain_board)
    drain_board()
    return list(s.stacks)


def _drain_forced_checks(s, drain_board) -> None:
    """Consume PokerKit's vestigial no-op checks so both sides end on a terminal.

    When every remaining opponent is all-in and the last live player owes
    nothing, that player has no decision: PokerKit reports them as the actor but
    offers exactly one action -- a check that closes the round and changes
    nothing. Our engine closes the round outright instead (`_round_has_no_decision`),
    so it stops emitting actions and PokerKit is left mid-street with chips still
    in `bets` and the board undealt. Comparing there compares a finished hand to
    an unfinished one.

    !! DIFFERENTIAL TOLERANCE -- read before widening !!

    This is the second place the harness forgives a PokerKit disagreement (the
    other is `_is_odd_chip_only`). Every such tolerance is a hole in the oracle,
    so the guard is deliberately the narrowest one that works: we advance ONLY
    when `check_or_call` is the SOLE legal action --

        s.can_check_or_call() and not s.can_fold() and not s.can_complete_bet_or_raise_to()

    -- i.e. only when PokerKit itself agrees the player has no choice to make.
    If a fold or a raise is still on offer, the player had a real decision our
    engine skipped, which is a genuine engine bug, and it stays a mismatch.

    Do NOT relax this to "actor is not None" or "engine is terminal". Either
    would silently absorb exactly the class of bug [E1] was -- a betting round
    the engine ends too early -- and that bug reached us precisely because the
    generator could not produce the states that expose it. A tolerance that
    swallows an engine's premature close-out would make the differential blind
    to its own most likely failure mode.
    """
    while (s.actor_index is not None and s.can_check_or_call()
           and not s.can_fold() and not s.can_complete_bet_or_raise_to()):
        s.check_or_call()
        drain_board()


def _pk_antes(setup: HandSetup) -> tuple[int, ...]:
    """Our ante model -> PokerKit's per-seat ante vector.

    THE ORACLE QUESTION, settled empirically before [A1] was touched.

    `ante_trimming_status` selects between two real rules variants, and neither
    one is right for both ante structures -- so the harness picks per hand.

    It does NOT change ante COLLECTION for a uniform ante (measured: identical
    contributions and pot under either flag, short payers included). What it
    changes is who contests the dead money at award time:

      * True  -- the ante pool is laddered. A player who could only cover part
                 of the ante staked only the bottom layer and does not contest
                 the rest.
      * False -- the whole ante pool is contested by everyone, including a
                 player who is all-in FROM the ante for less than it.

    Our engine ladders (see `_pots` and wave-3 [A1]), so True is the faithful
    oracle for uniform antes. Confirmed by hand, not just by green tests: for
    `short_stack_setup(16)` -- stacks (3615, 6, 173, 2148), ante 25, seat1
    all-in for 6 holding the BEST hand -- the laddered award gives seat1 only
    the 4x6=24 main pot, which is what our engine and PokerKit-True both
    produce; PokerKit-False hands seat1 an extra 57 by letting it contest an
    ante layer it never paid into.

    But True cannot be used with a SINGLE-PAYER ante, because trimming toward
    the smallest ante owed (zero, since nobody else owes one) trims the lone
    ante away entirely: PokerKit never collects it at all (measured: pot 150 vs
    250 on the same hand) and models a game with no ante. So bb-ante hands pass
    False, where the trimming rule has no lone ante to destroy.

    The two settings agree wherever both are defined, which is why this split is
    a faithful oracle rather than a convenient one: a single payer produces
    exactly one dead level, so there is no ante ladder for False to get wrong.

    THE HEADS-UP ROTATION TRAP. PokerKit's raw ante vector is rotated relative
    to seat index when n == 2: `antes=(100, 0)` charges seat 1, and
    `antes=(0, 100)` charges seat 0. (Its blinds rotate the same way -- HU the
    button posts the small blind -- which is why our seat convention already
    puts seat1=SB/button.) Getting this backwards silently charges the WRONG
    player and the differential then measures the wrong game, so n=2 is handled
    explicitly here rather than excluded: HU with a BB ante is a real final-table
    structure, and excluding it would rebuild exactly the kind of generator blind
    spot [E1] and [A2] were both caused by.
    """
    antes = [setup.ante] * len(setup.stacks)
    if setup.bb_ante:
        bb_seat = 0 if len(setup.stacks) == 2 else (setup.button + 2) % len(setup.stacks)
        idx = (bb_seat + 1) % 2 if len(setup.stacks) == 2 else bb_seat
        antes[idx] += setup.bb_ante
    return tuple(antes)


def _is_odd_chip_only(hand: Hand, mine: list[int], theirs: list[int]) -> bool:
    """True iff the only difference is odd-chip *placement* among tied winners.

    PokerKit's multiway all-in showdown merges side pots via an internal
    hand-killing/`can_win_now` heuristic, so in rare multiway all-in ties the
    single odd chip of a split can land on a different tied winner than our
    standard "lowest eligible seat" rule. That is a labelling difference, not a
    mechanics bug: winners, pot totals and every stack agree to within one chip
    that nets to zero, and only among players who hold an identical hand.
    """
    diffs = [m - t for m, t in zip(mine, theirs)]
    moved = [i for i, d in enumerate(diffs) if d != 0]
    if not moved or sum(diffs) != 0:
        return False
    # Odd chips are tiny: at most one per split pot (<= number of seats). A real
    # mis-split would move a whole half-pot, far above this cap.
    if any(abs(diffs[i]) > len(mine) for i in moved):
        return False
    board = hand.full_board[:5]
    grouped: dict[int, int] = {}  # hand rank -> net chips moved within that tie
    for i in moved:
        if hand.folded[i] or hand.hole[i] is None:
            return False  # a folded/non-showdown seat changed -> real bug
        rank = rank_showdown(hand.hole[i], board)  # type: ignore[arg-type]
        grouped[rank] = grouped.get(rank, 0) + diffs[i]
    # every reallocation stayed within one group of identically-ranked winners
    return all(net == 0 for net in grouped.values())


def check_hand(seed: int, action_fn=random_action,
               setup_fn=random_setup) -> tuple[str, str]:
    """Play + compare one seeded hand.

    Returns (status, detail): status is "exact" (stacks identical), "oddchip"
    (identical up to the documented PokerKit odd-chip-placement quirk), or
    "mismatch" (a real discrepancy — an engine bug).

    ``setup_fn`` selects the hand generator, so the same oracle can be pointed
    at a different slice of the state space (see `short_stack_setup`).
    """
    setup = setup_fn(seed)
    hand = play_engine(setup, seed, action_fn)
    mine = hand.final_stacks()
    # Chips are conserved or the engine is wrong in a way no oracle comparison
    # should have to discover for us. Checked here so a pot-formation bug is
    # named as such instead of surfacing as a confusing stack diff (wave-3 [A1]
    # shipped two ladder rewrites; one of them silently destroyed chips).
    if sum(mine) != sum(setup.stacks):
        return "mismatch", (
            f"seed={seed} CHIPS NOT CONSERVED: start={sum(setup.stacks)} "
            f"end={sum(mine)}\n  stacks0={setup.stacks} final={mine}\n"
            f"  contrib={hand.contrib} table_dead={hand.table_dead}"
        )
    theirs = pokerkit_final_stacks(setup, hand.action_history)
    if mine == theirs:
        return "exact", ""
    status = "oddchip" if _is_odd_chip_only(hand, mine, theirs) else "mismatch"
    detail = (
        f"seed={seed} n={len(setup.stacks)} ante={setup.ante}\n"
        f"  stacks0={setup.stacks}\n"
        f"  engine  ={mine}\n"
        f"  pokerkit={theirs}\n"
        f"  actions ={hand.action_history}"
    )
    return status, detail


def scan_chunk(bounds: tuple[int, int], action_fn=random_action,
               setup_fn=random_setup) -> tuple[int, int, list[str]]:
    """Worker: classify a [lo, hi) seed range -> (exact, oddchip, mismatches)."""
    import warnings

    warnings.filterwarnings("ignore")  # PokerKit's dealable-card notices
    lo, hi = bounds
    exact = odd = 0
    mismatches: list[str] = []
    for seed in range(lo, hi):
        status, detail = check_hand(seed, action_fn, setup_fn)
        if status == "exact":
            exact += 1
        elif status == "oddchip":
            odd += 1
        else:
            mismatches.append(detail)
    return exact, odd, mismatches


def run_differential(
    n: int, start: int = 0, workers: int | None = None, chunk: int = 400,
    action_fn=random_action, setup_fn=random_setup,
) -> tuple[int, int, list[str]]:
    """Run the PokerKit differential over ``n`` seeded hands across processes.

    ``action_fn`` (a module-level, picklable policy) lets callers drive the same
    oracle through a different code path — Slice H reuses this with a discrete
    env-encoding policy for the SingleEnvAdapter exit."""
    import concurrent.futures as cf
    import functools
    import os

    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    bounds = [(s, min(s + chunk, start + n)) for s in range(start, start + n, chunk)]
    exact = odd = 0
    mismatches: list[str] = []
    worker = functools.partial(scan_chunk, action_fn=action_fn, setup_fn=setup_fn)
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        for e, o, m in pool.map(worker, bounds):
            exact += e
            odd += o
            mismatches.extend(m)
    return exact, odd, mismatches
