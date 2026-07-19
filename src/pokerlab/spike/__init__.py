"""M7 spike: tiny-HUNL depth-limited value-net pipeline + M4 tier-2 activation.

    from pokerlab.spike.tier2 import (
        make_inline_solution_for, make_drain_solver, cache_solve, spotkey_str,
    )

`tier2` wires the Slice-D subgame solver into the HH grading pipeline through
Slice F's existing seams (no `hh` changes). `hunl` is the tiny-HUNL research
spike (data-gen → value net → depth-limited eval vs oracle); see
docs/notes/tiny-hunl-spike.md for the L4 go/no-go verdict.
"""
