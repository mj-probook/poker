.PHONY: test soak bench lint

test:            ## fast suite (<60s target; xdist — internal diff pools divide by worker count)
	uv run pytest -n 4

soak:            ## heavy differentials (1M-hand PokerKit differential etc.)
	uv run pytest -m slow -q

bench:           ## milestone-exit benchmarks (solver flops25 rivers; value-net heavy runs live in soak)
	uv run pytest -m bench -q --no-header

lint:
	uv run python -m compileall -q src
