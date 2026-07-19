"""Persistence layer (plan §3 storage; impl doc §2).

SQLite app state via the checked-in `schema.sql`; derived metrics (leak
rankings, accuracy trends) are QUERIES in `views.py`, never tables.

    db.connect(path) -> sqlite3.Connection      # schema applied
    db.insert_drill_attempt / upsert_sr_state / ...
    views.worst_leak_categories / attempt_history_trend / overall_accuracy
"""
