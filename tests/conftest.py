"""Test taxonomy, assigned by location so no test file needs decorators.

unit         default: pure-function / schema / contract tests (fast, no data, no network)
physics      invariants of the forward model and projector (adjoint, conservation, units, gradients)
regression   pinned numbers: frozen tables, kit facts, driver outputs on tiny fixtures
integration  needs an optional external stack (XCIST in .venv-xcist); skipped when absent
hpc          needs a live cluster; never in CI (none yet; cluster runs are covered by run records)

Passing any of these shows code matches its spec; scientific claims need registered run evidence.
"""

import pytest

_BY_PATH = (
    ("tests/physics/", "physics"),
    ("tests/data/", "regression"),
    ("tests/exp002/", "regression"),
    ("tests/p02/", "regression"),
    ("tests/p03/", "regression"),
    ("tests/aapm/test_consolidation_facts.py", "regression"),
    ("tests/paired/test_xcist.py", "integration"),
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        path = item.path.as_posix()
        kind = next((k for frag, k in _BY_PATH if frag in path), "unit")
        item.add_marker(getattr(pytest.mark, kind))
