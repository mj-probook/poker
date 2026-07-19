# Poker Lab

Personal poker lab: learn game theory + RL by building; train the author for live full-ring MTT NLHE. Single-user, local, non-commercial.

**Read before coding:**
- `docs/PLAN.md` — the authority on scope, milestones (M0–M7), exits, and rationale (rev 3, twice-reviewed).
- `/Users/mj/Documents/Obsidian/mj-probook/poker-lab.impl.md` — code-level contracts and per-slice TDD behavior lists.

**Hard rules:**
- `src/pokerlab/types.py` is the frozen shared contract — propose changes to the team lead; never drift it.
- Python 3.12 via uv only: `uv run pytest`, `uv add`. Never pip/system python.
- TDD, vertical: one failing test → minimal code → discover → repeat. No bulk test-writing, no speculative abstractions.
- Tests: `tests/test_<module>_*.py`, fixtures in `tests/fixtures/`, no network, fixed seeds. Fast suite <60s; heavy runs behind `slow`/`bench` markers (see Makefile).
- Grading honesty is load-bearing: tier 3 (multiway) NEVER reports ev_loss — only frequency-deviation flags. Nothing vendor-derived (GTO Wizard etc.) enters answer keys — in-house chart engine + own solves only.
- SQLite schema in `store/schema.sql`; derived metrics are queries in `store/views.py`, never tables.

**Module map** (plan §3): `engine` M0 · `cfr` M1 (+ exploitability utility) · `charts` M1.5 · `solver` M3 · `drills`+`web` M2 · `hh` M4 · `rebel` M5/M7 · `env` M6 · `store` persistence.
