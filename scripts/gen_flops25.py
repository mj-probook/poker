"""Generate the 25-flop benchmark fixture ``benchmarks/flops25.json`` (plan §1,
§8 M3; impl doc §3 Slice D).

The fixture pins a fixed HU postflop spot — **BTN open, BB call, 40bb** — and 25
canonical flops stratified across the 8 texture classes (solver/texture.py):
≥3 per class where the class has that many canonical flops, spread evenly within
each class for variety, padded to 25 from the largest class.

The preflop **ranges are FIXTURE CONSTANTS, not solver output** — a plausible,
self-authored 40bb Button-vs-BigBlind single-raised-pot configuration. Nothing
here is vendor-derived (CLAUDE.md hard rule); the construction rule is documented
inline below and is fully reproducible. Downstream solves consume these ranges;
they are never treated as an answer key themselves.

Spot arithmetic (heads-up, BTN = SB): SB 0.5 + BB 1.0 posted; BTN opens to 2.5bb,
BB calls 2.5bb → postflop **pot 5.0bb**, each has **37.5bb behind**. Bet grid is
33% / 75% / 125% pot + jam.

    uv run python scripts/gen_flops25.py
"""

from __future__ import annotations

import json
from pathlib import Path

from pokerlab.charts.hands import HAND_CLASSES, hand_ranks, is_pair
from pokerlab.engine.cards import cards_to_str
from pokerlab.solver import iso, texture

OUT = Path(__file__).resolve().parents[1] / "benchmarks" / "flops25.json"

STACK_BB = 40.0
OPEN_TO_BB = 2.5
POT_BB = 5.0          # both invest 2.5 preflop
EFF_STACK_BB = 37.5   # behind, postflop
BET_GRID = [0.33, 0.75, 1.25, "jam"]


def _btn_open() -> list[str]:
    """~50% Button opening range (documented rule).

    pairs: all (22+) · suited: A2s+, K4s+, Q6s+, J7s+, T7s+, and 4-9-high
    connectors/one-gappers (gap ≤ 2, low ≥ 4) · offsuit: A9o+, KTo+, QTo+, JTo.
    """
    keep = []
    for h in HAND_CLASSES:
        hi, lo = hand_ranks(h)
        suited = h.endswith("s")
        if is_pair(h):
            keep.append(h)
        elif suited:
            if hi == 14 or (hi == 13 and lo >= 4) or (hi == 12 and lo >= 6) \
                    or (hi == 11 and lo >= 7) or (hi == 10 and lo >= 7) \
                    or (hi <= 9 and hi - lo <= 2 and lo >= 4):
                keep.append(h)
        else:  # offsuit
            if (hi == 14 and lo >= 9) or (hi == 13 and lo >= 10) \
                    or (hi == 12 and lo >= 10) or (hi == 11 and lo >= 10):
                keep.append(h)
    return keep


def _bb_call() -> list[str]:
    """BB flat-call range vs a Button open (documented rule).

    Premiums (KK+, AKs/AKo, AQs) 3-bet and are *excluded* from this flat range.
    pairs: 22-QQ · suited: A2s-AJs, K5s+, Q7s+, J8s+, T8s+, 54s-98s connectors ·
    offsuit: ATo-AQo, KJo+, QJo, JTo.
    """
    keep = []
    for h in HAND_CLASSES:
        hi, lo = hand_ranks(h)
        suited = h.endswith("s")
        if is_pair(h):
            if hi <= 12:  # 22-QQ (KK/AA 3-bet)
                keep.append(h)
        elif suited:
            if (hi == 14 and lo <= 11) or (hi == 13 and lo >= 5) \
                    or (hi == 12 and lo >= 7) or (hi == 11 and lo >= 8) \
                    or (hi == 10 and lo >= 8) or (hi <= 9 and hi - lo == 1 and lo >= 4):
                keep.append(h)
        else:  # offsuit
            if (hi == 14 and 10 <= lo <= 12) or (hi == 13 and lo >= 11) \
                    or (hi == 12 and lo == 11) or (hi == 11 and lo == 10):
                keep.append(h)
    return keep


def _stratified_flops() -> list[dict]:
    """25 canonical flops, ≥3 per texture class where possible, evenly spread."""
    by_class: dict[str, list] = {t: [] for t in texture.TEXTURES}
    for flop in iso.all_canonical_flops():
        by_class[texture.texture(flop)].append(flop)

    chosen: list = []
    per_class = 3
    for t in texture.TEXTURES:
        flops = by_class[t]
        take = min(per_class, len(flops))
        # evenly spaced indices for variety, deterministic
        idxs = [round(i * (len(flops) - 1) / max(take - 1, 1)) for i in range(take)]
        for i in sorted(set(idxs)):
            chosen.append((t, flops[i]))

    # pad to 25 from the largest class (twotone_dry) with fresh evenly-spaced flops
    used = {f for _, f in chosen}
    pad_class = max(texture.TEXTURES, key=lambda t: len(by_class[t]))
    i = 0
    pad_flops = by_class[pad_class]
    while len(chosen) < 25 and i < len(pad_flops):
        f = pad_flops[len(pad_flops) - 1 - i]
        if f not in used:
            chosen.append((pad_class, f))
            used.add(f)
        i += 1

    return [
        {"cards": cards_to_str(f), "iso_class": iso.iso_class(f), "texture": t}
        for t, f in chosen[:25]
    ]


def main() -> None:
    btn, bb = _btn_open(), _bb_call()
    fixture = {
        "meta": {
            "formation": "BTNopen_BBcall",
            "stack_bb": STACK_BB,
            "open_to_bb": OPEN_TO_BB,
            "pot_bb": POT_BB,
            "eff_stack_bb": EFF_STACK_BB,
            "bet_grid": BET_GRID,
            "generated_by": "scripts/gen_flops25.py",
            "note": (
                "Ranges are fixture constants (self-authored 40bb BTNvsBB SRP), "
                "not solver output and not vendor-derived. See module docstring "
                "for the exact construction rule."
            ),
        },
        "ranges": {"BTN": sorted(btn), "BB": sorted(bb)},
        "flops": _stratified_flops(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, indent=2) + "\n")
    from collections import Counter
    dist = Counter(f["texture"] for f in fixture["flops"])
    print(f"wrote {OUT}")
    print(f"  BTN range: {len(btn)} classes   BB range: {len(bb)} classes")
    print(f"  25 flops across classes: {dict(dist)}")


if __name__ == "__main__":
    main()
