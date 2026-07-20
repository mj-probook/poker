"""Console entry points for the training loop (plan §5.3; wave-3 [P4']).

Two operations the loop is made of — import a session, drain the solve backlog —
were reachable only by writing Python against `hh.persist`. These are thin
argparse wrappers: every line here is argument handling, site routing, or
printing. No grading, scheduling or solving logic lives in this module, and none
should ever move here.

    pokerlab-import HAND_HISTORY...   parse -> grade -> persist -> summary
    pokerlab-batch                    drain the tier-2 queue once -> outcomes
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from pokerlab.hh.ggpoker import parse_ggpoker, split_ggpoker
from pokerlab.hh.model import ParsedHand
from pokerlab.hh.persist import drain_batch_queue, persist_session
from pokerlab.hh.pokerstars import parse_pokerstars, split_pokerstars
from pokerlab.hh.population import load_population
from pokerlab.hh.report import grade_session
from pokerlab.spike import tier2
from pokerlab.store import db

DEFAULT_DB = os.environ.get("POKERLAB_DB", "pokerlab.db")


def detect_site(raw: str) -> str | None:
    """The site that wrote this hand history, from its header, or None.

    Sniffing the text rather than the filename: the plan's v1 sites both write a
    fixed first token, and a file renamed on the way out of the client is still
    the same hand history.
    """
    head = raw.lstrip()[:64]
    if head.startswith("PokerStars Hand #"):
        return "PokerStars"
    if head.startswith("Poker Hand #"):
        return "GGPoker"
    return None


_PARSERS = {"PokerStars": parse_pokerstars, "GGPoker": parse_ggpoker}
# Keyed by the same site names `detect_site` returns, so routing and splitting
# cannot disagree about what a file is (round-4 finding [R2']).
_SPLITTERS = {"PokerStars": split_pokerstars, "GGPoker": split_ggpoker}


def _read_hands(paths: list[str]
                ) -> tuple[list[ParsedHand], list[str], list[str],
                           list[tuple[str, str, str]]]:
    """(parsed, raws, problems, lost) — an unreadable file never costs the others.

    `lost` is the subset of problems that are LOST HANDS: a real hand history,
    from a site we recognize, that our parser could not read. Those are the M4
    promise's subject (plan §8: "never silently dropped") and the caller records
    them durably — printing alone left no trace by morning on a cron run, which
    is the very gap `failed_hands` exists to close, reappearing one layer up in
    the entry point.

    The other two problems are deliberately NOT lost hands, because their
    remedies differ and collapsing remedies is its own defect:
      * unreadable file — there is no text to store and nothing to re-import;
        the remedy is fixing permissions or the path, not the parser.
      * unrecognized format — we cannot even name a site for it, and nothing
        was lost; the remedy is "check what you passed".
    """
    parsed: list[ParsedHand] = []
    raws: list[str] = []
    problems: list[str] = []
    lost: list[tuple[str, str, str]] = []      # (site, raw, reason)
    for p in paths:
        name = Path(p).name
        try:
            raw = Path(p).read_text()
        except OSError as exc:
            problems.append(f"{name}: unreadable ({exc.strerror})")
            continue
        site = detect_site(raw)
        if site is None:
            problems.append(f"{name}: unrecognized hand-history format")
            continue
        # Split first: a real auto-saved file holds hundreds of hands, and
        # parsing it as one hand merged them all into the first hand's record
        # (round-4 finding [R2']). Isolation moves from per-FILE to per-HAND —
        # the machinery below is unchanged and was always right, it was just
        # being fed one giant hand.
        chunks = _SPLITTERS[site](raw)
        for i, chunk in enumerate(chunks, 1):
            where = name if len(chunks) == 1 else f"{name} hand {i}/{len(chunks)}"
            try:
                parsed.append(_PARSERS[site](chunk))
                # The CHUNK, not the file: `raws` is the per-hand re-import
                # payload and the dedup key's source. Storing the whole session
                # under every hand would make one bad hand un-re-importable
                # without re-importing all of them.
                raws.append(chunk)
            except Exception as exc:  # noqa: BLE001 - per-hand isolation boundary
                reason = f"{type(exc).__name__}: {exc}"
                problems.append(f"{where}: {reason}")
                # A header-level failure now reaches `failed_hands` instead of
                # dying on stdout (round-4 finding [P2]): it is a chunk of a
                # recognized site's file, so `site` is known and the row is
                # honest — no invented "unknown" site is needed, because a file
                # we cannot attribute never gets here (see `detect_site` above).
                lost.append((site, chunk, reason))
    return parsed, raws, problems, lost


def main_import(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="pokerlab-import",
        description="Import hand histories: parse, grade by tier, persist.")
    ap.add_argument("files", nargs="+", help="PokerStars or GGPoker HH files")
    ap.add_argument("--db", default=DEFAULT_DB, help="SQLite path")
    args = ap.parse_args(argv)

    parsed, raws, problems, lost = _read_hands(args.files)
    for p in problems:
        print(f"  SKIPPED {p}")

    conn = db.connect(args.db)
    at = datetime.now(timezone.utc).isoformat()

    # Record lost hands BEFORE the empty-session exit. A cron run whose files
    # are all broken is exactly the case where the failure must survive, and
    # returning early here would have left the worst run with no trace at all.
    for site, raw, reason in lost:
        db.insert_failed_hand(conn, site, raw, reason, at)

    if not parsed:
        print(f"no hands to import ({len(lost)} recorded as failed)")
        return 0

    report = grade_session(parsed, population=load_population())
    counts = persist_session(
        conn, parsed, report, raw_texts=raws, graded_at=at)

    print(f"\n=== session: {len(parsed)} hands from {len(args.files)} files ===")
    print(f"imported   : {counts['imported']} ({counts['skipped']} already known)")
    # `routed` counts every decision the grader saw; `graded` counts the ones
    # that got an answer key and were persisted. They differ by the pending
    # backlog, so printing them as one number would hide exactly that gap.
    print(f"routed     : {report.total} decisions  {report.routed} by tier")
    print(f"graded     : {counts['graded']} with a reference")
    # Two distinct failure stages, both durable, both counted here: a hand can
    # die at PARSE (never becomes a ParsedHand) or later at GRADING (replay or
    # reconcile rejects it). Reporting only the second was the entry point's
    # share of the M4 gap.
    print(f"failed     : {report.failed_hand_count + len(lost)} hands could "
          f"not be graded ({len(lost)} at parse, "
          f"{report.failed_hand_count} at grading)")
    for site, _raw, reason in lost:
        print(f"   - {site} (parse): {reason}")
    for fh in report.failed_hands:
        print(f"   - hand #{fh.hand_index} ({fh.hand_id}): {fh.reason}")
    # A pending count is not a footnote: plan §5.3 marks a session PARTIAL until
    # its tier-2 backlog is solved, so the operator must see it at import time.
    print(f"pending    : {len(report.pending)} decisions "
          f"({counts['queued']} queued for solve)")
    print(f"drillable  : {counts['seeded']} leak categories seeded")
    if counts["queued"]:
        print("\nsession is PARTIAL — run `pokerlab-batch` to solve the backlog")
    return 0


def main_batch(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="pokerlab-batch",
        description="Drain the tier-2 solve queue once and report outcomes.")
    ap.add_argument("--db", default=DEFAULT_DB, help="SQLite path")
    ap.add_argument("--iters", type=int, default=tier2.DEFAULT_ITERS,
                    help="CFR iterations per solve")
    # The report tells the operator "fix the bug, then retry the batch"; the
    # drain only picks up 'pending', so without this that advice needed Python.
    # Defaults to 'failed' exactly as `db.retry_failed_batch` does — the CLI
    # must not quietly widen it, because retrying an 'unsolvable' or
    # 'mismatched' row re-fails by construction and sends the operator round
    # the loop the report exists to warn about.
    ap.add_argument("--retry", nargs="*", metavar="STATUS",
                    help="reopen terminal rows before draining (default: "
                         "failed; pass e.g. 'unsolvable' after a solver "
                         "upgrade, or 'mismatched' after re-importing)")
    args = ap.parse_args(argv)

    conn = db.connect(args.db)
    if args.retry is not None:
        statuses = tuple(args.retry) or ("failed",)
        try:
            reopened = db.retry_failed_batch(conn, statuses)
        except ValueError as exc:
            ap.error(str(exc))          # exits 2 — a bad status is a usage error
        print(f"reopened    : {reopened} ({', '.join(statuses)})")
    result = drain_batch_queue(
        conn, tier2.make_drain_solver(conn, iters=args.iters),
        graded_at=datetime.now(timezone.utc).isoformat())

    print("=== batch drain ===")
    # Every outcome is printed, including the ones that are not failures: a
    # 'missed' row is normal backlog and stays drainable, an 'unsolvable' one is
    # terminal. Collapsing them is what lost real backlog in wave 2 [E15].
    for k in ("done", "missed", "unsolvable", "failed", "recovered",
              "mismatched"):
        if k in result:
            print(f"{k:<12}: {result[k]}")
    print(f"still queued: {len(db.pending_batch(conn))}")
    return 0
