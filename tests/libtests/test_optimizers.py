import pytest
import numpy as np
from api.lib.optimizers import *
from api.lib.preprocessing.initializers import random_normal, ones

w_trainable_1 = ones(2, 3)
b_trainable_1 = ones(1, 3)

w_trainable_2 = ones(3, 3)
b_trainable_2 = ones(1, 3)

trainable_nodes = [
    (w_trainable_1, b_trainable_1),
    (w_trainable_2, b_trainable_2)
]


@pytest.mark.parametrize('trainable', trainable_nodes, ids=['2x3', '3x3'])
class TestGradientDescent:

    def test_compute_gradients(self, session, trainable):
        optimizer = GradientDescent(
            lr=1,
            trainable_variables=trainable,
            session=session
        )

        W, b = trainable
        X = random_normal(W.value.shape[1], W.value.shape[0])
        op = X @ W + b

        session.run(op)

        received, _ = optimizer.compute_gradients(op)
        assert np.array_equal(received[0], session.run(X).T)  # df/dw
        assert received[1] == 1  # df/db

    @pytest.mark.parametrize('lr', [0.01, 2], ids=['lr=0.01', 'lr=2'])
    def test_minimize(self, session, lr, trainable, mocker):
        optimizer = GradientDescent(
            lr=lr,
            trainable_variables=trainable,
            session=session
        )

        W, b = trainable
        w_init_value, b_init_value = W.value.copy(), b.value.copy()
        X = random_normal(W.value.shape[1], W.value.shape[0])
        op = X @ W + b

        mocker.patch.object(
            optimizer,
            'compute_gradients',
            return_value=(
                [np.ones(W.value.shape), np.ones(b.value.shape)], None
            )
        )
        optimizer.minimize(op)

        w_expected = w_init_value - np.ones(W.value.shape) * lr
        assert np.array_equal(W.value, w_expected)

        b_expected = b_init_value - np.ones(b.value.shape) * lr
        assert np.array_equal(b.value, b_expected)
