"""Explicit placeholders for opt-in hardware qualification suites.

Each backend gets an adapter and measured result in its own qualification kata.
Keeping these visible (and skipped) prevents ordinary CI from silently pretending
that its deterministic CPU contract test exercised accelerator hardware.
"""

import pytest

pytestmark = pytest.mark.accelerator


@pytest.mark.parametrize("backend", ["cuda", "mps", "coreml", "mlx"])
def test_backend_qualification_is_not_part_of_ordinary_ci(backend):
    pytest.skip(f"{backend} benchmark adapter is opt-in and not implemented in baseline csk2")
