-- Poker lab persistence (plan §3; impl doc §2).
-- Derived metrics (leak rankings, EV-loss trends, skill gates) are QUERIES
-- in store/views.py — never tables.

CREATE TABLE IF NOT EXISTS imported_hands(
  id INTEGER PRIMARY KEY,
  site TEXT NOT NULL,
  raw TEXT NOT NULL,
  parsed_json TEXT NOT NULL,
  imported_at TEXT NOT NULL,
  hand_uid TEXT,                      -- site hand number from the HH header
  -- Re-importing a file must not double-count hands into the leak stats.
  -- NULLs stay distinct in SQLite, so hands whose number did not parse are
  -- simply never deduped (they import, as before).
  UNIQUE(site, hand_uid)
);

CREATE TABLE IF NOT EXISTS gradings(
  id INTEGER PRIMARY KEY,
  hand_id INTEGER NOT NULL REFERENCES imported_hands(id),
  decision_idx INTEGER NOT NULL,
  tier INTEGER NOT NULL CHECK (tier IN (1, 2, 3)),
  chosen TEXT NOT NULL,
  best TEXT NOT NULL,                 -- '' when the tier has no single best action
                                      -- (tier 3 never names one) — see hh/persist.py
  ev_loss REAL,                       -- NULL iff tier = 3 (enforced in code + trigger)
  -- Canonical 4-part category key owned by drills/categories.py:
  --   formation|street|action|depth
  -- depth is a snapped bucket (5/8/10/15/20) for preflop jam/fold and '-' for
  -- postflop; formation takes a '.icm' suffix for ICM variants.
  leak_key TEXT NOT NULL,
  graded_at TEXT NOT NULL
);

-- Grading honesty is load-bearing (plan §5.3): tier 3 must never carry ev_loss.
CREATE TRIGGER IF NOT EXISTS gradings_tier3_no_evloss
BEFORE INSERT ON gradings
WHEN NEW.tier = 3 AND NEW.ev_loss IS NOT NULL
BEGIN
  SELECT RAISE(ABORT, 'tier-3 gradings must not report ev_loss');
END;

CREATE TABLE IF NOT EXISTS drill_attempts(
  id INTEGER PRIMARY KEY,
  -- Same canonical 4-part key as gradings.leak_key (see above) — that identity
  -- is the whole point: an HH-detected leak selects the drill that trains it.
  leak_key TEXT NOT NULL,
  kind TEXT NOT NULL,
  chosen TEXT NOT NULL,
  correct INTEGER NOT NULL,
  ev_loss REAL NOT NULL,
  ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sr_state(
  leak_key TEXT PRIMARY KEY,
  easiness REAL NOT NULL,
  interval_days REAL NOT NULL,
  reps INTEGER NOT NULL,
  due TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS solution_index(
  spot_key TEXT PRIMARY KEY,
  path TEXT NOT NULL,
  solver_version TEXT NOT NULL,
  exploitability REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS batch_queue(
  id INTEGER PRIMARY KEY,
  -- Coarse queue label `formation|street` (hh/persist.py:spot_key) — NOT a
  -- types.SpotKey, which also carries stack and board buckets. It is handed to
  -- the drain solver alongside the re-derived decision, which is what actually
  -- carries the board/pot/stack, so the label only has to group the backlog.
  -- (This is now the ONLY 'spot_key' in the schema that is not a types.SpotKey;
  -- drill_attempts.spot_key was the third meaning and is now leak_key.)
  spot_key TEXT NOT NULL,
  hand_id INTEGER NOT NULL,
  decision_idx INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'done', 'failed'))
);
