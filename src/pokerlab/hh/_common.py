"""Shared hand-history parsing for the PokerStars and GGPoker text formats.

Both sites emit the same body grammar (seat list, blind/ante posts, per-street
action lines, board reveals, showdown, collected + SUMMARY totals); only the
header line differs. `parse_hand` takes a site-specific header regex (which
must expose named groups ``hid, tid, level, sb, bb``) and does the rest.

Blind/ante posts are folded into the contribution ledger but never emitted as
engine actions — the engine posts them from `HandSetup`; the action list holds
only voluntary actions in table order. Board cards are read incrementally: FLOP
carries three cards in its sole bracket, TURN/RIVER echo the prior board and
carry the new card in a *second* bracket.
"""

from __future__ import annotations

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
        return 0, next(amt for _, amt, is_bb in posts if is_bb)
    seats = {seat for seat, _, _ in posts}
    if len(seats) == 1 and posts[0][0] == _bb_poster(lines, idx):
        return 0, posts[0][1]
    return posts[0][1], 0


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


def parse_hand(text: str, *, site: str, header_re: re.Pattern) -> ParsedHand:
    lines = [ln.rstrip("\n") for ln in text.strip().splitlines()]

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

    setup = HandSetup(
        stacks=tuple(stacks),
        button=button,
        bb=bb,
        sb=sb,
        ante=ante,
        hole=tuple(hole),
        board=tuple(board),
        bb_ante=bb_ante,
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
