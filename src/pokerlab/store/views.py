"""Derived-metric queries over the store (impl doc §2; plan §3, §5.3).

Leak rankings and accuracy trends are computed here as QUERIES — never
persisted as tables (plan §3: "derived metrics are queries, not tables").
Slice F reuses this module for HH leak reports over `gradings`.
"""

from __future__ import annotations

import sqlite3


def worst_leak_categories(conn: sqlite3.Connection, limit: int = 5,
                          min_attempts: int = 1) -> list[dict]:
    """Drill leak_keys ranked by error rate (worst first).

    error_rate = fraction of attempts marked incorrect. Ties break toward the
    more-practised category (more attempts -> higher confidence in the rate).
    `min_attempts` filters out categories with too little data to rank.
    """
    rows = conn.execute(
        "SELECT leak_key,"
        "       COUNT(*)                AS attempts,"
        "       SUM(1 - correct)        AS errors,"
        "       1.0 - AVG(correct)      AS error_rate"
        "  FROM drill_attempts"
        " GROUP BY leak_key"
        " HAVING COUNT(*) >= ?"
        " ORDER BY error_rate DESC, attempts DESC, leak_key"
        " LIMIT ?",
        (min_attempts, limit),
    )
    return [dict(r) for r in rows]


def attempt_history_trend(conn: sqlite3.Connection) -> list[dict]:
    """Per-attempt running accuracy over time (a learning-curve query)."""
    rows = conn.execute(
        "SELECT ts, correct,"
        "       AVG(correct) OVER ("
        "         ORDER BY ts, id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
        "       ) AS cumulative_accuracy"
        "  FROM drill_attempts"
        " ORDER BY ts, id"
    )
    return [dict(r) for r in rows]


def overall_accuracy(conn: sqlite3.Connection) -> float:
    """Fraction of all drill attempts marked correct (0.0 if none)."""
    row = conn.execute(
        "SELECT AVG(correct) AS acc FROM drill_attempts").fetchone()
    return float(row["acc"]) if row["acc"] is not None else 0.0


# --------------------------------------------------------------------------- #
# HH leak reports over `gradings` (Slice F; plan §5.3).
#   * tiers 1–2 carry a trustworthy ev_loss -> ranked by EV-loss/100.
#   * tier 3 has NO ev_loss (honesty rule) -> listed SEPARATELY by action
#     frequency, never mixed into the EV-loss ranking.
# --------------------------------------------------------------------------- #
def hh_leak_report(conn: sqlite3.Connection, limit: int = 5) -> list[dict]:
    """Top leak_keys by EV-loss/100, over tiers 1–2 only (tier 3 excluded).

    ev_loss_per_100 = mean bb conceded per decision × 100 (a bb/100 rate).
    Ties break toward more decisions (higher confidence), then leak_key.
    """
    rows = conn.execute(
        "SELECT leak_key,"
        "       COUNT(*)                 AS decisions,"
        "       SUM(ev_loss)             AS total_ev_loss,"
        "       AVG(ev_loss)             AS avg_ev_loss,"
        "       100.0 * AVG(ev_loss)     AS ev_loss_per_100"
        "  FROM gradings"
        " WHERE tier IN (1, 2) AND ev_loss IS NOT NULL"
        " GROUP BY leak_key"
        " ORDER BY ev_loss_per_100 DESC, decisions DESC, leak_key"
        " LIMIT ?",
        (limit,),
    )
    return [dict(r) for r in rows]


def category_priority(conn: sqlite3.Connection) -> list[dict]:
    """Every category either source knows about, with BOTH training signals.

    The leak->drill join (wave-3 [P1']). The SM-2 scheduler used to rank what to
    resurface by drill error-rate alone, which meant a leak the HH import had
    just priced in EV was invisible to it until the drill loop independently
    stumbled on that category — so "my hands pick my drills" (plan §5.3) never
    actually held. This is the union of both vocabularies:

      * `ev_loss_per_100` — bb/100 conceded in real hands, tiers 1-2 ONLY. Tier
        3 carries no ev_loss (honesty rule), so it is excluded rather than
        counted as 0 and ranked against tiers that have a trustworthy oracle.
      * `error_rate` — fraction of drill attempts answered wrong.

    A category missing from one source scores 0 there, never NULL, so callers
    can sort on either field without special-casing. Derived, so it is a query
    and never a table (plan §3).
    """
    rows = conn.execute(
        "WITH hh AS ("
        "  SELECT leak_key, COUNT(*) AS decisions,"
        "         100.0 * AVG(ev_loss) AS ev_loss_per_100"
        "    FROM gradings WHERE tier IN (1, 2) AND ev_loss IS NOT NULL"
        "   GROUP BY leak_key),"
        " dr AS ("
        "  SELECT leak_key, COUNT(*) AS attempts,"
        "         1.0 - AVG(correct) AS error_rate"
        "    FROM drill_attempts GROUP BY leak_key),"
        " keys AS (SELECT leak_key FROM hh UNION SELECT leak_key FROM dr)"
        "SELECT k.leak_key,"
        "       COALESCE(hh.decisions, 0)          AS decisions,"
        "       COALESCE(hh.ev_loss_per_100, 0.0)  AS ev_loss_per_100,"
        "       COALESCE(dr.attempts, 0)           AS attempts,"
        "       COALESCE(dr.error_rate, 0.0)       AS error_rate"
        "  FROM keys k"
        "  LEFT JOIN hh ON hh.leak_key = k.leak_key"
        "  LEFT JOIN dr ON dr.leak_key = k.leak_key"
        " ORDER BY error_rate DESC, ev_loss_per_100 DESC, k.leak_key"
    )
    return [dict(r) for r in rows]


def tier3_frequency_report(conn: sqlite3.Connection) -> list[dict]:
    """Tier-3 (multiway) decisions grouped by leak_key with counts — the
    frequency signal, listed separately from the EV-loss ranking. No ev_loss
    is ever surfaced here (it is NULL for tier 3 by construction)."""
    rows = conn.execute(
        "SELECT leak_key, COUNT(*) AS decisions"
        "  FROM gradings"
        " WHERE tier = 3"
        " GROUP BY leak_key"
        " ORDER BY decisions DESC, leak_key"
    )
    return [dict(r) for r in rows]
