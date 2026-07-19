"""Persistence layer (plan §3 storage; impl doc §2).

SQLite app state via the checked-in `schema.sql`; derived metrics (leak
rankings, accuracy trends) are QUERIES in `views.py`, never tables.

`db` holds the helpers for every app table — the drill loop
(`drill_attempts`, `sr_state`), grading (`gradings`), HH import
(`imported_hands`, deduped on site+hand_uid), and the tier-2 library and
backlog (`solution_index`, `batch_queue`).

    db.connect(path) -> sqlite3.Connection      # schema applied
    db.insert_drill_attempt / upsert_sr_state / insert_imported_hand / ...
    db.index_solution / enqueue_batch / recover_running_batch
    views.worst_leak_categories / attempt_history_trend / overall_accuracy
"""
