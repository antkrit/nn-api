import pytest

from api.core.utils import atleast_3d


@pytest.mark.parametrize("array", [1, [1], [[1, 2]], [[[1, 2]]], [[[[1, 2]]]]])
def test_at_least3d(array):
    assert atleast_3d(array).ndim >= 3
