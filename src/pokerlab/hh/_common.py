"""Shared hand-history parsing for PokerStars and GGPoker (Slice F; impl doc §3).

Both sites emit the same body grammar (seat list, blind/ante posts, per-street
action lines, board reveals, showdown, collected + SUMMARY totals); only the
header line differs. `parse_hand` takes a site-specific header regex (which
must expose named groups ``hid, tid, level, sb, bb``) and does the rest.

Blind/ante posts are folded into the contribution ledger but never emitted as
engine actions — the engine posts them from `HandSetup`; the action list holds
only voluntary actions in table order. Board cards are read incrementally: FLOP
carries three cards in its sole bracket, TURN/RIVER echo the prior board and
carry the new card in a *second* bracket.

Two table conventions this module is responsible for getting right, because a
wrong answer to either silently corrupts the replay rather than failing:

* **Ante structure** (`_read_antes`) — a per-player ante is charged to EVERY
  seat while a big-blind ante is one payment for the table, so misreading one
  as the other mis-states the pot n-fold. The rule is **BB-ante iff exactly one
  seat posted an ante AND that seat is the one posting the big blind**; a lone
  ante from anyone else is per-player. Keying on the poster's identity rather
  than the number of posters is what makes this correct at both heads-up and
  3-handed (round-1 finding [12], round-2 finding [E6]).

* **Dead button** (`_button_index`) — when a player busts, the announced button
  can land on an unoccupied seat. The fallback is the nearest occupied seat
  counter-clockwise, which keeps the engine's derived SB/BB aligned with the
  blinds the history actually posts (round-1 finding [9]).
"""

from __future__ import annotations

import dataclasses
import re

from pokerlab.engine.cards import card_from_str
from pokerlab.engine.state import HandSetup
from pokerlab.hh.model import ParsedHand
from pokerlab.types import Action

_TABLE = re.compile(r"Table '(?P<name>[^']+)' (?P<max>\d+)-max Seat #(?P<btn>\d+) is the button")
_SEAT = re.compile(r"Seat (?P<no>\d+): (?P<name>.+?) \((?P<stack>\d+) in chips\)")
_ANTE = re.compile(r"(?P<name>.+?): posts (?:the )?ante (?P<amt>\d+)")
# GGPoker spells the BB ante out; PokerStars emits a lone plain ante line.
_BB_ANTE = re.compile(r"(?P<name>.+?): posts (?:the )?big blind ante (?P<amt>\d+)")
_SB = re.compile(r"(?P<name>.+?): posts small blind (?P<amt>\d+)")
_BB = re.compile(r"(?P<name>.+?): posts big blind (?P<amt>\d+)")
_DEALT = re.compile(r"Dealt to (?P<name>.+?) \[(?P<cards>[^\]]+)\]")
_STREET = re.compile(
    r"\*\*\* (?P<street>FLOP|TURN|RIVER) \*\*\*.*?\[(?P<cards>[^\]]+)\]"
    r"(?:\s*\[(?P<extra>[^\]]+)\])?"
)
_UNCALLED = re.compile(r"Uncalled bet \((?P<amt>\d+)\) returned to (?P<name>.+)")
_COLLECTED = re.compile(r"(?P<name>.+?) collected (?P<amt>\d+) from (?:main pot|side pot|pot)")
_TOTAL = re.compile(r"Total pot (?P<amt>\d+)")
_SHOWS = re.compile(r"(?P<name>.+?): shows \[(?P<cards>[^\]]+)\]")

_A_FOLD = re.compile(r"(?P<name>.+?): folds")
_A_CHECK = re.compile(r"(?P<name>.+?): checks")
_A_CALL = re.compile(r"(?P<name>.+?): calls (?P<amt>\d+)")
_A_BET = re.compile(r"(?P<name>.+?): bets (?P<amt>\d+)")
_A_RAISE = re.compile(r"(?P<name>.+?): raises (?P<by>\d+) to (?P<to>\d+)")


def _cards(s: str) -> tuple[int, ...]:
    return tuple(card_from_str(tok) for tok in s.split())


def _ante_posts(lines: list[str], idx: dict[str, int]) -> list[tuple[int, int, bool]]:
    """Every ante line as (seat, amount, is_bb_ante), in file order."""
    out: list[tuple[int, int, bool]] = []
    for ln in lines:
        bb = _BB_ANTE.match(ln)
        if bb and bb["name"] in idx:
            out.append((idx[bb["name"]], int(bb["amt"]), True))
            continue
        a = _ANTE.match(ln)
        if a and a["name"] in idx:
            out.append((idx[a["name"]], int(a["amt"]), False))
    return out


def _replayed(setup: HandSetup, actions: list[tuple[int, Action]],
              uncalled: int) -> tuple[int, tuple[int, ...]] | None:
    """(contested pot, final stacks) the engine produces, or None if unreplayable."""
    from pokerlab.engine.state import Hand      # local: avoids an import cycle

    try:
        hand = Hand(setup)
        for seat, action in actions:
            if hand.to_act != seat:
                return None
            hand.apply(action)
        if not hand.is_terminal():
            return None
        return sum(hand.contrib) - uncalled, tuple(hand.final_stacks())
    except Exception:                            # noqa: BLE001 - probe only
        return None


def _arbitrate_ante(setup: HandSetup, actions: list[tuple[int, Action]],
                    stated_pot: int, uncalled: int,
                    stated_final: tuple[int, ...]) -> HandSetup:
    """Let the stated Total pot settle an ambiguous ante reading (finding [H5]).

    "N posts the ante X" is textually ambiguous between a per-player ante and a
    modern big-blind ante, and the poster's identity — the rule `_read_antes`
    uses — is only decisive when the file names the big blind as the poster.
    Where it is not, the SUMMARY's Total pot arbitrates: the two readings put
    different amounts of dead money in, so at most one of them reproduces it.

    The match required is the FULL reconciliation criterion — stated pot AND
    stated final stacks — not the pot alone. The two readings put the same dead
    money in from *different seats*, so a pot-only test could flip to a reading
    that reproduces the total while charging the wrong player. Matching stacks
    too means a flip can only ever produce a hand that fully reconciles.

    Only ever flips a reading that does NOT reconcile to one that does, so a
    hand the identity rule already got right can never be moved. If neither
    reading reconciles the original is kept and reconciliation drops the hand
    loudly, which is the correct outcome for a hand we cannot model.
    """
    if not (setup.ante or setup.bb_ante) or not stated_pot:
        return setup
    want = (stated_pot, stated_final)
    if _replayed(setup, actions, uncalled) == want:
        return setup                             # identity rule already agrees
    amount = setup.ante or setup.bb_ante
    flipped = dataclasses.replace(
        setup,
        ante=0 if setup.ante else amount,
        bb_ante=amount if setup.ante else 0,
    )
    if _replayed(flipped, actions, uncalled) == want:
        return flipped
    return setup


def _bb_poster(lines: list[str], idx: dict[str, int]) -> int | None:
    """Engine seat index of whoever posts the big blind, if the text says."""
    for ln in lines:
        m = _BB.match(ln)
        if m and m["name"] in idx:
            return idx[m["name"]]
    return None


def _read_antes(lines: list[str], idx: dict[str, int]) -> tuple[int, int]:
    """(per_player_ante, bb_ante) — exactly one of the two is non-zero.

    Distinguishing them matters a lot: the engine charges a per-player ante to
    EVERY seat, so reading a big-blind ante as a per-player one overcharges the
    pot n-fold (round-1 finding [12], and this is the target sites' modern
    format). The rules:

      * several ante lines        -> per-player ante (the classic format);
      * explicit "big blind ante" -> BB ante, whatever the count;
      * a lone ante line posted BY THE BIG BLIND -> BB ante;
      * a lone ante line from anyone else        -> per-player ante.

    The discriminator is *who* posted, not how many seats did (round-2 finding
    [E6]). Counting seats got both edges wrong: it read a lone per-player ante
    at a 3-handed table as a BB ante, and — because it excluded n==2 outright —
    charged both seats for the lone ante line that modern PokerStars emits for
    a heads-up big-blind ante, which dropped every such hand at reconciliation.
    A big-blind ante is posted by the big blind by definition, so asking that
    question directly settles both without a seat-count special case.
    """
    posts = _ante_posts(lines, idx)
    if not posts:
        return 0, 0
    if any(is_bb for _, _, is_bb in posts):
        return 0, max(amt for _, amt, is_bb in posts if is_bb)
    # The ante is the amount everyone OWES, so it is the maximum posted, not the
    # first line (round-3 finding [H1]). A player too short to pay in full posts
    # a partial ante and is all-in; if that line happens to come first — which it
    # does whenever the short stack sits in an earlier seat — reading posts[0]
    # takes the *shortfall* as the table ante and undercharges every other seat.
    # The engine already clamps each seat to its stack (`_post_ante`), so passing
    # the full ante is what makes the short stack come out right too.
    amount = max(amt for _, amt, _ in posts)
    seats = {seat for seat, _, _ in posts}
    if len(seats) == 1 and posts[0][0] == _bb_poster(lines, idx):
        return 0, amount
    return amount, 0


def _button_index(seat_nos: list[int], button_seat: int) -> int:
    """Engine seat index for the announced button seat — dead button tolerated.

    When a player busts, the button can be announced on a seat nobody occupies
    (the "dead button"): the blinds stay where the rotation put them and the
    button is simply not in play. Falling back to the nearest occupied seat
    **counter-clockwise** (the largest occupied seat number below the announced
    one, wrapping to the highest seat) is what keeps the engine's derived
    SB=(button+1)%n / BB=(button+2)%n consistent with the blinds the history
    actually posts — verified against tests/fixtures/hh/ps_dead_button.txt
    (round-1 finding [9]).
    """
    if button_seat in seat_nos:
        return seat_nos.index(button_seat)
    earlier = [s for s in seat_nos if s < button_seat]
    return seat_nos.index(max(earlier) if earlier else max(seat_nos))


def split_hands(text: str, boundary: str) -> list[str]:
    """Split a session file into one chunk per hand (round-4 finding [R2']).

    `boundary` is the site's hand-header PREFIX ("PokerStars Hand #"), not its
    full header regex, and that distinction is load-bearing: a hand whose
    header is malformed still starts a chunk, so it can be isolated and
    recorded as a failed hand. Splitting on the strict header instead would
    silently glue a corrupt hand onto its predecessor — reintroducing the very
    merge this exists to fix, in the case least likely to be noticed.

    Leading text before the first boundary (a client preamble, or junk) is
    returned as its own chunk rather than dropped: the caller isolates and
    records it, and discarding unparseable input is what plan §8 forbids.

    BYTE-FAITHFUL: the chunks concatenate back to `text` exactly. This is a
    contract, not an implementation detail, because `failed_hands` is keyed on
    the stored raw text and a re-import must reproduce it to clear the row. A
    first version joined `splitlines()` with "\\n", which silently normalized
    CRLF and dropped the trailing newline — so no chunk was ever byte-identical
    to its source file, and a failure recorded before this change could never
    be cleared again. The report would then claim forever that a hand is broken
    which now imports fine, which is the exact defect `clear_failed_hand`
    exists to prevent.
    """
    if not text.strip():
        return []
    # Character offsets, so slices preserve the original bytes verbatim —
    # including line endings and the trailing newline.
    offsets, pos = [], 0
    for ln in text.splitlines(keepends=True):
        offsets.append(pos)
        pos += len(ln)
    starts = [offsets[i] for i, ln in enumerate(text.splitlines())
              if ln.startswith(boundary)]
    if not starts:
        return [text]
    bounds = ([0] if starts[0] != 0 else []) + starts + [len(text)]
    chunks = [text[a:b] for a, b in zip(bounds, bounds[1:])]
    # A whitespace-only leading chunk is MERGED FORWARD, not dropped. Dropping
    # it silently lost those bytes, so a file with a leading blank line failed
    # the byte-faithfulness contract above and its pre-split failure row could
    # not clear. Merging keeps the bytes without manufacturing a junk chunk
    # that would fail to parse and record a failed hand made of whitespace.
    if len(chunks) > 1 and not chunks[0].strip():
        chunks[1] = chunks[0] + chunks[1]
        chunks = chunks[1:]
    return [c for c in chunks if c.strip()]


def peek_hand_uid(text: str, header_re: re.Pattern) -> str | None:
    """The hand number from a chunk's header line, or None if it won't parse.

    A probe for the failure path: a hand can die in its BODY (no table line,
    desynced actions) long after its header parsed perfectly, and that failure
    should be recorded under the number the file gave it rather than as
    "unidentified" while the number sits in the stored raw text (round-4
    finding [R1b]).

    Deliberately reuses the site's own `header_re` rather than a second pattern
    describing what a hand number looks like — two homes for that would drift,
    and the drift would be silent. Reads `lines[0]` ONLY, exactly as
    `parse_hand` does, so peek and parse can never disagree about which line is
    the header or what it says.

    Returns None rather than raising, and None is a MEANINGFUL answer: a hand
    whose header itself is malformed genuinely has no number we can attribute,
    and inventing one would be worse than silence.
    """
    lines = text.strip().splitlines()
    if not lines:
        return None
    m = header_re.search(lines[0])
    return m["hid"] if m else None


def parse_hand(text: str, *, site: str, header_re: re.Pattern,
               boundary: str | None = None) -> ParsedHand:
    lines = [ln.rstrip("\n") for ln in text.strip().splitlines()]

    # Reject plural input rather than merging it (round-4 finding [R2']).
    # Pre-fix, a 2-hand file matched the header on line 0 and then ran the body
    # grammar over every subsequent hand's lines too, returning ONE hand with
    # the first hand's id and all 21 actions — no exception raised. Adding a
    # plural entry point alone would have left that trap armed for every
    # existing caller, so the singular contract is enforced where it is stated.
    if boundary is not None:
        n = sum(1 for ln in lines if ln.startswith(boundary))
        if n > 1:
            raise ValueError(
                f"{site}: expected a single hand, got {n} hands — "
                f"use the file-level parser to split a session file")

    m = header_re.search(lines[0])
    if not m:
        raise ValueError(f"not a {site} hand header: {lines[0]!r}")
    sb, bb = int(m["sb"]), int(m["bb"])
    hand_id, tid, level = m["hid"], m["tid"], m["level"]

    tbl = next((_TABLE.search(ln) for ln in lines if _TABLE.search(ln)), None)
    if not tbl:
        raise ValueError(f"missing {site} table line")
    button_seat = int(tbl["btn"])

    seats: list[tuple[int, str, int]] = []
    for ln in lines:
        s = _SEAT.match(ln)
        if s:
            seats.append((int(s["no"]), s["name"], int(s["stack"])))
    seats.sort(key=lambda t: t[0])
    seat_nos = [t[0] for t in seats]
    names = [t[1] for t in seats]
    stacks = [t[2] for t in seats]
    n = len(seats)
    idx = {name: i for i, name in enumerate(names)}
    button = _button_index(seat_nos, button_seat)

    ante, bb_ante = _read_antes(lines, idx)

    contributed = [0] * n
    collected = [0] * n
    street_commit = [0] * n
    hole: list[tuple[int, int] | None] = [None] * n
    board: list[int] = []
    actions: list[tuple[int, Action]] = []
    uncalled = 0
    total_pot = 0
    hero = -1

    # antes + blinds -> contribution ledger only (engine re-posts from setup)
    for seat, amt, _is_bb in _ante_posts(lines, idx):
        contributed[seat] += amt
    for ln in lines:
        for rx in (_SB, _BB):
            bm = rx.match(ln)
            if bm and bm["name"] in idx:
                i = idx[bm["name"]]
                contributed[i] += int(bm["amt"])
                street_commit[i] = int(bm["amt"])

    in_actions = False
    for ln in lines:
        if ln.startswith("*** HOLE CARDS"):
            in_actions = True
            continue
        d = _DEALT.match(ln)
        if d and d["name"] in idx:
            hero = idx[d["name"]]
            hole[hero] = _cards(d["cards"])  # type: ignore[assignment]
            continue
        st = _STREET.match(ln)
        if st:
            street_commit = [0] * n
            board.extend(_cards(st["extra"] if st["extra"] else st["cards"]))
            continue
        sh = _SHOWS.match(ln)
        if sh and sh["name"] in idx:
            hole[idx[sh["name"]]] = _cards(sh["cards"])  # type: ignore[assignment]
            continue
        u = _UNCALLED.match(ln)
        if u and u["name"].strip() in idx:
            amt = int(u["amt"])
            uncalled += amt
            contributed[idx[u["name"].strip()]] -= amt
            continue
        c = _COLLECTED.match(ln)
        if c and c["name"] in idx:
            collected[idx[c["name"]]] += int(c["amt"])
            continue
        t = _TOTAL.search(ln)
        if t:
            total_pot = int(t["amt"])
            continue
        if in_actions:
            _parse_action(ln, idx, street_commit, contributed, actions)

    setup = _arbitrate_ante(
        HandSetup(
            stacks=tuple(stacks),
            button=button,
            bb=bb,
            sb=sb,
            ante=ante,
            hole=tuple(hole),
            board=tuple(board),
            bb_ante=bb_ante,
        ),
        actions=actions,
        stated_pot=total_pot,
        uncalled=uncalled,
        stated_final=tuple(stacks[i] - contributed[i] + collected[i]
                           for i in range(n)),
    )
    return ParsedHand(
        site=site,
        setup=setup,
        actions=actions,
        seat_names=tuple(names),
        hero=hero,
        contributed=tuple(contributed),
        collected=tuple(collected),
        uncalled=uncalled,
        total_pot=total_pot,
        hand_id=hand_id,
        tournament_id=tid,
        level=level,
    )


def _parse_action(ln, idx, street_commit, contributed, actions) -> None:
    r = _A_RAISE.match(ln)
    if r and r["name"] in idx:
        i = idx[r["name"]]
        to = int(r["to"])
        contributed[i] += to - street_commit[i]
        street_commit[i] = to
        actions.append((i, ("raise", to)))
        return
    b = _A_BET.match(ln)
    if b and b["name"] in idx:
        i = idx[b["name"]]
        contributed[i] += int(b["amt"])
        street_commit[i] += int(b["amt"])
        actions.append((i, ("bet", street_commit[i])))
        return
    c = _A_CALL.match(ln)
    if c and c["name"] in idx:
        i = idx[c["name"]]
        contributed[i] += int(c["amt"])
        street_commit[i] += int(c["amt"])
        actions.append((i, ("call", street_commit[i])))
        return
    ck = _A_CHECK.match(ln)
    if ck and ck["name"] in idx:
        actions.append((idx[ck["name"]], ("check", 0)))
        return
    f = _A_FOLD.match(ln)
    if f and f["name"] in idx:
        actions.append((idx[f["name"]], ("fold", 0)))
        return
