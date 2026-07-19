"""Slice D — TexasSolver export + optional diff (impl doc §3 Slice D note).

The exporter is always tested; the actual TexasSolver run/diff is skipped unless
a binary is provided via the ``TEXASSOLVER_BIN`` environment variable (external
binary, not present in CI).
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "compare_texassolver",
    Path(__file__).resolve().parents[1] / "scripts" / "compare_texassolver.py",
)
compare = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare)

from pokerlab.solver.flops25 import load_flops25


def test_exporter_emits_wellformed_texassolver_input():
    fx = load_flops25()
    entry = fx["flops"][0]
    text = compare.texassolver_input(entry, fx, iters=123)
    # board rendered comma-separated, e.g. "As,Kd,7h"
    assert f"set_board {compare._board_commas(entry['cards'])}" in text
    assert "set_pot 5.0" in text
    assert "set_effective_stack 37.5" in text
    # ranges carried through (OOP=BB, IP=BTN)
    assert "set_range_oop " in text and "set_range_ip " in text
    assert fx["ranges"]["BB"][0] in text and fx["ranges"]["BTN"][0] in text
    # bet grid fractions -> percentages
    assert "33,75,125" in text
    assert "set_max_iteration 123" in text
    assert text.strip().endswith("dump_result output.json")


def _binary():
    return os.environ.get("TEXASSOLVER_BIN") or shutil.which("console_solver")


@pytest.mark.skipif(_binary() is None, reason="no TexasSolver binary available")
def test_texassolver_consumes_export(tmp_path):  # pragma: no cover - needs binary
    fx = load_flops25()
    out = tmp_path / "spot.txt"
    out.write_text(compare.texassolver_input(fx["flops"][0], fx, iters=50))
    compare.run_and_diff(_binary(), out)
