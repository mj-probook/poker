"""Derived-metric queries over the store (impl doc §2; plan §3, §5.3).

Leak rankings and accuracy trends are computed here as QUERIES — never
persisted as tables (plan §3: "derived metrics are queries, not tables").
Slice F reuses this module for HH leak reports over `gradings`.
"""

from __future__ import annotations

import sqlite3


def worst_leak_categories(conn: sqlite3.Connection, limit: int = 5,
                          min_attempts: int = 1) -> list[dict]:
    """Drill spot_keys ranked by error rate (worst first).

    error_rate = fraction of attempts marked incorrect. Ties break toward the
    more-practised category (more attempts -> higher confidence in the rate).
    `min_attempts` filters out categories with too little data to rank.
    """
    rows = conn.execute(
        "SELECT spot_key,"
        "       COUNT(*)                AS attempts,"
        "       SUM(1 - correct)        AS errors,"
        "       1.0 - AVG(correct)      AS error_rate"
        "  FROM drill_attempts"
        " GROUP BY spot_key"
        " HAVING COUNT(*) >= ?"
        " ORDER BY error_rate DESC, attempts DESC, spot_key"
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
