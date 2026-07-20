"""Export a flops25 spot to TexasSolver's input format and (optionally) diff
root strategies against a TexasSolver build (impl doc §3 Slice D note).

TexasSolver's CPU console reads a line-oriented command file (``set_pot``,
``set_board``, ``set_range_oop/ip``, ``set_bet_sizes``, ``build_tree``,
``start_solve``, ``dump_result``). This script emits that file for any of the 25
benchmark flops. If a TexasSolver binary path is supplied it runs the solve and
diffs the flop root strategy against our own solver; without a binary it just
writes the input file (and the accompanying test is skipped).

    uv run python scripts/compare_texassolver.py --flop 0 --out spot.txt
    uv run python scripts/compare_texassolver.py --flop 0 --binary /path/to/console_solver
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from pokerlab.engine.cards import card_to_str
from pokerlab.solver.flops25 import load_flops25


def _board_commas(cards_str: str) -> str:
    return ",".join(cards_str[i:i + 2] for i in range(0, len(cards_str), 2))


def texassolver_input(entry: dict, fixture: dict, iters: int = 200,
                      accuracy: float = 0.005) -> str:
    """A TexasSolver command file for one flops25 flop (OOP=BB, IP=BTN)."""
    meta = fixture["meta"]
    oop = ",".join(fixture["ranges"]["BB"])   # BB is out of position
    ip = ",".join(fixture["ranges"]["BTN"])   # BTN is in position
    fracs = [str(int(f * 100)) for f in meta["bet_grid"] if isinstance(f, (int, float))]
    sizes = ",".join(fracs)
    lines = [
        f"set_pot {meta['pot_bb']}",
        f"set_effective_stack {meta['eff_stack_bb']}",
        f"set_board {_board_commas(entry['cards'])}",
        f"set_range_oop {oop}",
        f"set_range_ip {ip}",
        f"set_bet_sizes oop,flop,bet,{sizes}",
        f"set_bet_sizes oop,flop,raise,{sizes}",
        f"set_bet_sizes ip,flop,bet,{sizes}",
        f"set_bet_sizes ip,flop,raise,{sizes}",
        "set_allin_threshold 0.67",
        "build_tree",
        "set_thread_num 4",
        f"set_accuracy {accuracy}",
        f"set_max_iteration {iters}",
        "start_solve",
        "set_dump_rounds 1",
        "dump_result output.json",
    ]
    return "\n".join(lines) + "\n"


def run_and_diff(binary: str, input_file: Path) -> None:  # pragma: no cover - needs binary
    """Run TexasSolver on the exported input; print where output landed.

    A full root-strategy L1 diff against our own solver is left to the caller —
    it requires a bench-scale local flop solve (see tests/test_solver_flops25 /
    `make bench`). This wrapper verifies the binary consumes our export.
    """
    proc = subprocess.run([binary, "-i", str(input_file)], capture_output=True, text=True)
    print(proc.stdout[-2000:])
    if proc.returncode != 0:
        print("TexasSolver returned non-zero:", proc.returncode)
        print(proc.stderr[-2000:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flop", type=int, default=0, help="flops25 index 0..24")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("texassolver_spot.txt"))
    ap.add_argument("--binary", type=str, default=None, help="TexasSolver console binary")
    args = ap.parse_args()

    fx = load_flops25()
    entry = fx["flops"][args.flop]
    text = texassolver_input(entry, fx, iters=args.iters)
    args.out.write_text(text)
    print(f"wrote TexasSolver input for flop {args.flop} ({entry['cards']}, "
          f"{entry['texture']}) -> {args.out}")
    if args.binary:
        run_and_diff(args.binary, args.out)
    else:
        print("no --binary given: export only (run a TexasSolver console on the file)")


if __name__ == "__main__":
    main()
