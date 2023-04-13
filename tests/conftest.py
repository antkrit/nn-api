import numpy as np
import pytest

import api.core.autograd as ag


@pytest.fixture
def graph():
    with ag.Graph() as g:
        yield g


@pytest.fixture
def session(graph):
    return ag.Session()


UNARY_TEST_CASES = [
    np.random.randint(1, 10),
    np.random.randint(1, 10, size=(3,)),
    np.random.randint(1, 10, size=(2, 3)),
]
BINARY_TEST_CASES = [
    (np.random.randint(1, 10), np.random.randint(1, 10)),
    (np.random.randint(1, 10), np.random.randint(1, 10, size=(3,))),
    (np.random.randint(1, 10), np.random.randint(1, 10, size=(3, 3))),
    (np.random.randint(1, 10, size=(3,)), np.random.randint(1, 10, size=(3,))),
    (
        np.random.randint(1, 10, size=(3, 3)),
        np.random.randint(1, 10, size=(3, 3)),
    ),
    (
        np.random.randint(1, 10, size=(4, 1, 2)),
        np.random.randint(1, 10, size=(2, 2)),
    ),
    (
        np.random.randint(1, 10, size=(1, 2)),
        np.random.randint(1, 10, size=(5, 2, 2)),
    ),
]


@pytest.fixture(params=UNARY_TEST_CASES)
def test_case_unary(request):
    return request.param


@pytest.fixture(params=BINARY_TEST_CASES)
def test_case_binary(request):
    return request.param
