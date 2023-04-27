import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.core.autograd as ag
from api.core.generic import Model
from api.core.layers import Dense
from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def load_custom_model_sgd_mse(mocker):
    """Create model and patch model loading.

    Model created with parameters:
    - optimizer: gradient_descent
    - loss: mean_squared_error
    - metrics: null
    - layers: Input(shape=(1, 784)) -> Flatten() -> output
    """
    model = Model(input_shape=[1, 784])
    model.add(Dense(5))

    model.compile("gradient_descent", "mean_squared_error")
    mocker.patch("api.model.wrapper.Model._load_model", return_value=model)

    os.environ["MODEL_PATH"] = "_"

    return model


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
