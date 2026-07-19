.PHONY: test soak bench lint

test:            ## fast suite (<60s target)
	uv run pytest

soak:            ## heavy differentials (1M-hand PokerKit differential etc.)
	uv run pytest -m slow -q

bench:           ## milestone-exit benchmarks (full flops25, value-net evals)
	uv run pytest -m bench -q --no-header

lint:
	uv run python -m compileall -q src
