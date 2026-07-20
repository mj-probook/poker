"""The differential's internal pool must share the box with pytest-xdist.

The recorded wrinkle (wave-2 levers note): differential tests spawn
``cpu_count - 2`` workers internally, so running the suite under xdist would
oversubscribe the machine unless the internal pool is divided by the number
of xdist workers. xdist announces itself to workers via
``PYTEST_XDIST_WORKER_COUNT``.
"""

import os

from tests.diff_harness import _default_workers


def test_default_workers_unchanged_without_xdist(monkeypatch):
    monkeypatch.delenv("PYTEST_XDIST_WORKER_COUNT", raising=False)
    assert _default_workers() == max(1, (os.cpu_count() or 2) - 2)


def test_default_workers_divides_by_xdist_worker_count(monkeypatch):
    monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", "4")
    assert _default_workers() == max(1, ((os.cpu_count() or 2) - 2) // 4)


def test_default_workers_never_drops_below_one(monkeypatch):
    monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", "9999")
    assert _default_workers() == 1
