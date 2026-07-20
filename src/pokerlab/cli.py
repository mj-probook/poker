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

from pokerlab.hh.ggpoker import parse_ggpoker
from pokerlab.hh.model import ParsedHand
from pokerlab.hh.persist import drain_batch_queue, persist_session
from pokerlab.hh.pokerstars import parse_pokerstars
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


def _read_hands(paths: list[str]) -> tuple[list[ParsedHand], list[str], list[str]]:
    """(parsed, raws, problems) — an unreadable file never costs the others."""
    parsed: list[ParsedHand] = []
    raws: list[str] = []
    problems: list[str] = []
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
        try:
            parsed.append(_PARSERS[site](raw))
            raws.append(raw)
        except Exception as exc:  # noqa: BLE001 - per-file isolation boundary
            problems.append(f"{name}: {type(exc).__name__}: {exc}")
    return parsed, raws, problems


def main_import(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="pokerlab-import",
        description="Import hand histories: parse, grade by tier, persist.")
    ap.add_argument("files", nargs="+", help="PokerStars or GGPoker HH files")
    ap.add_argument("--db", default=DEFAULT_DB, help="SQLite path")
    args = ap.parse_args(argv)

    parsed, raws, problems = _read_hands(args.files)
    for p in problems:
        print(f"  SKIPPED {p}")
    if not parsed:
        print("no hands to import")
        return 0

    conn = db.connect(args.db)
    report = grade_session(parsed, population=load_population())
    counts = persist_session(
        conn, parsed, report, raw_texts=raws,
        graded_at=datetime.now(timezone.utc).isoformat())

    print(f"\n=== session: {len(parsed)} hands from {len(args.files)} files ===")
    print(f"imported   : {counts['imported']} ({counts['skipped']} already known)")
    # `routed` counts every decision the grader saw; `graded` counts the ones
    # that got an answer key and were persisted. They differ by the pending
    # backlog, so printing them as one number would hide exactly that gap.
    print(f"routed     : {report.total} decisions  {report.routed} by tier")
    print(f"graded     : {counts['graded']} with a reference")
    print(f"failed     : {report.failed_hand_count} hands could not be graded")
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
