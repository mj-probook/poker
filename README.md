# Poker Lab

A personal, local, single-user poker lab. Two goals, one codebase: **learn game
theory and RL by building them** (CFR family, depth-limited search with learned
value functions), and **train myself** for live full-ring MTT no-limit hold'em.

Not a product: no accounts, no serving, no polish, non-commercial. The authority
on scope, milestones, and rationale is **[`docs/PLAN.md`](docs/PLAN.md)**; this
file just tells you how to run it.

## Quickstart

Python 3.12 via [uv](https://docs.astral.sh/uv/) only — never pip or system
python.

```bash
uv sync                     # install deps
uv run pytest               # fast suite (target <60s)
make soak                   # heavy differentials (1M-hand PokerKit run, etc.)
make bench                  # milestone-exit benchmarks (solver flops25 rivers; value-net heavy runs are in soak)
```

Run the drill UI:

```bash
uv run pokerlab-web         # http://127.0.0.1:8000
```

| env var | default | meaning |
|---|---|---|
| `POKERLAB_DB` | `pokerlab.db` | SQLite file for drills, gradings, SR state |
| `POKERLAB_HOST` | `127.0.0.1` | bind address |
| `POKERLAB_PORT` | `8000` | port |

## Module map

| module | milestone | what it is |
|---|---|---|
| `engine` | M0 | hand state machine, betting, side pots, showdown; differential-tested vs PokerKit |
| `cfr` | M1 | CFR / CFR+ / MCCFR on Kuhn & Leduc, plus the best-response & exploitability utility everything else is scored with |
| `charts` | M1.5 | Malmuth-Harville ICM + jam/fold Nash solver — the in-house answer keys |
| `solver` | M3 | heads-up postflop subgame solver: vectorized CFR+ over 1326-combo ranges, suit isomorphism, texture buckets |
| `drills` | M2 | drill generation, decision-ε scoring, SM-2 spaced repetition |
| `web` | M2 | local FastAPI drill loop + a no-build static front end |
| `hh` | M4 | PokerStars/GGPoker hand-history parsers → tiered grading → leak report |
| `rebel` | M5 | toy ReBeL on Leduc: public belief states, depth-limited CFR, value net |
| `spike` | M7 | tiny-HUNL depth-limited value-net research spike |
| `env` | M6 | vectorized environment + exact best-response exploiter |
| `store` | — | SQLite persistence; derived metrics are queries (`store/views.py`), never tables |

## Status — honest version

Milestones M0–M7 have landed as vertical slices, with these deviations worth
knowing before trusting a number:

- **The M6 vectorized env is numpy, not Rust.** The plan schedules Rust for the
  M3 solver hot loop and the M6 env; neither port has happened. Everything is
  pure Python/numpy, so the solver is correct but slow — reaching the accuracy
  bar on a *full flop* subgame is ~an hour per flop, which is exactly why the
  plan schedules that port.
- **The M7 tiny-HUNL spike is a recorded NO-GO.** See
  [`docs/notes/tiny-hunl-spike.md`](docs/notes/tiny-hunl-spike.md) for the
  numbers and the reasoning. Nothing downstream depends on it — M4 tier-2
  grading uses exact cached solves, never the value net.
- **Grading honesty is enforced, not just documented.** Tier 3 (genuine
  multiway postflop) never reports an EV loss — only frequency-deviation flags —
  because no trustworthy oracle exists there. This is enforced in the code *and*
  by a SQLite trigger.
- **Nothing vendor-derived is in here.** Answer keys come from the in-house
  chart engine and our own solves only (see `CLAUDE.md`).

## Repo conventions

See [`CLAUDE.md`](CLAUDE.md): `src/pokerlab/types.py` is a frozen shared
contract, tests live in `tests/test_<module>_*.py` with fixed seeds and no
network, and heavy runs sit behind the `slow` / `bench` markers.
