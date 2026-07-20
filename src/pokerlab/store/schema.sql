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

-- A hand that could not be parsed or replayed. PLAN §8 M4 promises these are
-- "isolated in failed_hands and surfaced in the session summary — never
-- silently dropped"; round-1 [8] delivered the isolation as an in-memory
-- dataclass, which vanishes with the process. On the nightly-cron path nobody
-- reads stdout, so by morning a failed hand left no trace at all.
--
-- Deliberately NOT in `imported_hands`: a failed hand must not claim its
-- (site, hand_uid), or the dedup would make the failure permanent and the hand
-- could never be re-imported once the parser is fixed (wave-2 [E16]/[E17]).
-- That is also why `raw` is stored here — it is the whole re-import path.
CREATE TABLE IF NOT EXISTS failed_hands(
  id INTEGER PRIMARY KEY,
  site TEXT NOT NULL,
  hand_uid TEXT,                      -- site hand number IF the parse got that far
  raw TEXT NOT NULL,                  -- re-import after a parser fix needs this
  reason TEXT NOT NULL,               -- what went wrong, for the session summary
  imported_at TEXT NOT NULL           -- groups one import batch
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
  -- postflop; formation takes a '.icm' suffix for ICM variants. The depth field
  -- also carries a snapped per-player ante bucket when the ante is nonzero
  -- ('10a0.125', '10a0.25' — buckets {0, 0.125, 0.25} bb/player, wave-3 [E49]);
  -- a zero ante appends nothing, so ante-free keys are unchanged by that rev
  -- and existing rows still join.
  leak_key TEXT NOT NULL,
  graded_at TEXT NOT NULL,
  -- The frequency the reference assigns the hero's action, and the deviation
  -- flags raised against it. For tiers 1-2 that is the CHOSEN action's own
  -- frequency (context for the ev_loss); for tier 3 it is the POPULATION
  -- frequency, and these two columns are the entire honest output of that tier
  -- (plan §5.3: no EV-loss number, only frequency-deviation flags). The grader
  -- computed both from the start and persist discarded them (wave-3 [P3']).
  -- `flags` is a comma-joined list, '' when nothing was flagged.
  frequency REAL,
  flags TEXT NOT NULL DEFAULT '',
  -- A decision is graded once. Two concurrent drains can both see a batch row
  -- 'pending', both re-derive it and both grade it; without this the duplicate
  -- lands silently and double-counts into every leak statistic downstream
  -- (wave-3 [B3]). The drain relies on this to fail loudly instead.
  UNIQUE(hand_id, decision_idx)
);

-- Grading honesty is load-bearing (plan §5.3): tier 3 must never carry ev_loss.
-- Guarded on BOTH writes: an INSERT-only trigger holds the rule exactly until
-- someone writes an UPDATE, which is how R2 [E23] came back as wave-3 [B4].
CREATE TRIGGER IF NOT EXISTS gradings_tier3_no_evloss
BEFORE INSERT ON gradings
WHEN NEW.tier = 3 AND NEW.ev_loss IS NOT NULL
BEGIN
  SELECT RAISE(ABORT, 'tier-3 gradings must not report ev_loss');
END;

CREATE TRIGGER IF NOT EXISTS gradings_tier3_no_evloss_update
BEFORE UPDATE ON gradings
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
  -- Terminal causes are SEPARATE states, because they need separate operator
  -- guidance and the report is a DB read (wave-3, w3-product-builder):
  --   'failed'      a bug failed this row. Fix the bug, retry_failed_batch.
  --   'unsolvable'  structurally outside what the solver models. Will not
  --                 change without a solver upgrade -- not a transient fault.
  --   'mismatched'  the queue no longer describes the hand (the re-derived
  --                 decision is not the one queued, see hh/persist [B2]).
  -- Collapsing these to 'failed' made the report call a transient cause
  -- permanent. All three are reopenable via retry_failed_batch, but it reopens
  -- 'failed' by default: retry is the remedy only for that one. Reopening a
  -- 'mismatched' row without re-importing re-derives the same wrong spot and
  -- re-fails forever.
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'running', 'done', 'failed', 'unsolvable', 'mismatched'))
);
