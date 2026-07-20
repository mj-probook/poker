"""Freeze the jam/fold Nash reference table (impl doc §3 Slice C).

Derives the SB-jam and BB-call frequencies at 5/8/10/15/20bb from the in-house
chip-EV solver and writes tests/fixtures/jamfold_reference.json. The frozen
table is a REGRESSION baseline: test_charts_jamfold_reference asserts the
current solver still agrees with it on ≥99% of hands.

The `_meta.hand_audited` block records ~30 canonical entries that were manually
checked against well-known published heads-up Nash push/fold facts (the
equilibrium charts reproduced by HoldemResources / SnapShove and standard
push/fold theory). Nothing vendor-derived enters the table — it is 100% our own
solve; the published facts are only a sanity cross-reference (CLAUDE.md rule).

    uv run python scripts/gen_jamfold_reference.py
"""

from __future__ import annotations

import json
from pathlib import Path

from pokerlab.charts import hands
from pokerlab.charts.jamfold import solve_jamfold

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "jamfold_reference.json"
DEPTHS = (5, 8, 10, 15, 20)

# ~30 entries hand-audited against published HU Nash push/fold facts.
# Each: (hand, position, expected action per depth, human-readable fact).
#
# The `expect` map is what makes the audit load-bearing rather than decorative
# (round-1 finding [3]): the test asserts the SOLVER takes each stated action at
# each stated depth. It is transcribed from the published fact, NOT read back
# out of our own table — an entry only lists the depths its fact actually
# claims. Actions: SB -> "jam"|"fold", BB -> "call"|"fold".


def _all(action: str) -> dict[int, str]:
    return {d: action for d in DEPTHS}


def _until(shallow: str, deep: str, cutoff: int) -> dict[int, str]:
    """`shallow` below `cutoff` bb, `deep` at and above it."""
    return {d: (shallow if d < cutoff else deep) for d in DEPTHS}


HAND_AUDITED = [
    ("AA", "SB", _all("jam"), "premium jams at every depth 5-20bb"),
    ("KK", "SB", _all("jam"), "premium jams 5-20bb"),
    ("QQ", "SB", _all("jam"), "premium jams 5-20bb"),
    ("JJ", "SB", _all("jam"), "jams 5-20bb"),
    ("TT", "SB", _all("jam"), "jams 5-20bb"),
    ("22", "SB", _all("jam"), "any pair jams at <=20bb heads-up"),
    ("55", "SB", _all("jam"), "any pair jams 5-20bb"),
    ("99", "SB", _all("jam"), "any pair jams 5-20bb"),
    ("A2s", "SB", _all("jam"), "suited aces jam 5-20bb"),
    ("AKs", "SB", _all("jam"), "jams 5-20bb"),
    ("A5s", "SB", _all("jam"), "suited wheel ace jams 5-20bb"),
    ("A2o", "SB", _all("jam"), "offsuit aces jam wide heads-up (5-20bb)"),
    ("K9o", "SB", _all("jam"), "broadway-ish Kxo jams 5-20bb"),
    ("T9s", "SB", _all("jam"), "suited connector jams 5-20bb"),
    ("98s", "SB", _all("jam"), "suited connector jams 5-20bb"),
    ("54s", "SB", _all("jam"), "suited connector jams 5-20bb"),
    ("K2s", "SB", _until("jam", "fold", 20), "marginal suited K jams shallow, folds ~20bb"),
    ("Q2s", "SB", _until("jam", "fold", 15), "weak suited Q jams shallow, folds by 15bb"),
    ("J8o", "SB", _until("jam", "fold", 15), "weak Jxo jams shallow, folds by 15bb"),
    ("72o", "SB", _all("fold"), "worst hand: folds at 5-20bb (only jams sub-4bb)"),
    ("32o", "SB", _all("fold"), "trash offsuit folds 5-20bb"),
    ("AA", "BB", _all("call"), "always calls a jam"),
    ("KK", "BB", _all("call"), "always calls a jam"),
    ("QQ", "BB", _all("call"), "always calls a jam"),
    ("33", "BB", _all("call"), "small pair calls 5-20bb"),
    ("55", "BB", _all("call"), "calls 5-20bb"),
    # this fact speaks only to a 10bb jam, so only 10bb is audited
    ("A2s", "BB", {10: "call"}, "A2s calling a 10bb SB jam is a (close) call"),
    ("AKo", "BB", _all("call"), "calls 5-20bb"),
    ("KTo", "BB", _all("call"), "calls 5-20bb"),
    ("QTs", "BB", _all("call"), "calls 5-20bb"),
    ("22", "BB", _until("call", "fold", 20), "small pair calls shallow, folds vs a 20bb jam"),
    ("A2o", "BB", _until("call", "fold", 20), "calls shallow, folds vs a 20bb jam"),
    ("72o", "BB", _all("fold"), "never calls (folds 5-20bb)"),
]


def main() -> None:
    sb_jam: dict[str, dict[str, float]] = {}
    bb_call: dict[str, dict[str, float]] = {}
    for d in DEPTHS:
        sol = solve_jamfold(float(d))
        sb_jam[str(d)] = {h: round(float(sol.sb_jam[i]), 4)
                          for i, h in enumerate(hands.HAND_CLASSES)}
        bb_call[str(d)] = {h: round(float(sol.bb_call[i]), 4)
                           for i, h in enumerate(hands.HAND_CLASSES)}

    payload = {
        "_meta": {
            "description": "In-house HU push/fold Nash reference (SB jam/fold vs "
                           "BB call/fold). Chip-EV, blinds 0.5/1, no ante. Frozen "
                           "regression baseline for the chart engine.",
            "solver": "pokerlab.charts.jamfold.solve_jamfold (CFR+, own equity matrix)",
            "equity_matrix": "src/pokerlab/charts/data/equity169.npz (seed 1234, 20k trials)",
            "depths_bb": list(DEPTHS),
            "provenance": "100% self-generated; published Nash facts used only to "
                          "sanity-audit, never as data (CLAUDE.md).",
            "hand_audited": [
                {"hand": h, "position": pos,
                 "expect": {str(d): a for d, a in expect.items()},
                 "published_fact": fact}
                for h, pos, expect, fact in HAND_AUDITED
            ],
            "agreement_threshold": 0.99,
        },
        "sb_jam": sb_jam,
        "bb_call": bb_call,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KiB); {len(HAND_AUDITED)} audited entries")


if __name__ == "__main__":
    main()
