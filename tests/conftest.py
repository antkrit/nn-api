import pytest
import numpy as np
import api.lib.autograd as ag


@pytest.fixture
def session():
    return ag.Session()


@pytest.fixture
def graph():
    with ag.Graph() as g:
        yield g


UNARY_TEST_CASES = [
    4,
    [5],
    np.array([1, 2, 3]),
    [[1, 2, 3], [1, 2, 3]]
]
BINARY_TEST_CASES = [
    (-4, 5),
    (2, [2, 3, 4]),
    (-7, np.array([[2, 3, 4], [2, 3, 4]])),
    (np.array([1, -2, 3]), [2, 3, 4]),
    ([[1, -2, 3], [-1, 2, 3]], np.array([[2, 3, 4], [2, 3, 4]])),
]


@pytest.fixture(params=UNARY_TEST_CASES)
def test_case_unary(request):
    return request.param


@pytest.fixture(params=BINARY_TEST_CASES)
def test_case_binary(request):
    return request.param
