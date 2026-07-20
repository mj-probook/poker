"""Shared pytest fixtures."""

import pytest


@pytest.fixture(scope="session")
def leduc_sigma_star():
    """A converged full-game Leduc CFR+ solution (tree, average profile).

    Loaded from a checked-in artifact rather than re-solved. 700 CFR+ iterations
    on Leduc is deterministic to the bit, so the 4.0s solve was pure
    recomputation — 7% of the fast suite's budget spent rederiving a constant.
    Regenerate with `scripts/gen_leduc_sigma_star.py`.

    The tree is rebuilt (0.01s) rather than stored, so the profile can never be
    paired with a stale tree.

    NashConv is re-checked on load. That guard is what makes caching safe: the
    profile underwrites M5 exit assertions, so if a change to CFR+ or Leduc
    invalidates the artifact this fails loudly here, at the fixture, instead of
    quietly weakening every test that builds on it. It costs 0.008s.
    """
    import json
    from pathlib import Path

    from pokerlab.cfr.exploit import nash_conv
    from pokerlab.cfr.game import build_tree
    from pokerlab.cfr.leduc import LeducPoker

    path = Path(__file__).parent / "fixtures" / "leduc_sigma_star.json"
    payload = json.loads(path.read_text())
    profile = {k: {int(a): p for a, p in d.items()}
               for k, d in payload["profile"].items()}

    tree = build_tree(LeducPoker())
    nc = nash_conv(tree, profile)
    assert nc < 1e-3, (
        f"checked-in leduc_sigma_star is stale: NashConv {nc:.6g} >= 1e-3. "
        "CFR+ or Leduc changed under it — regenerate with "
        "`uv run python scripts/gen_leduc_sigma_star.py`."
    )
    return tree, profile
