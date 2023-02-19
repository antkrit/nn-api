import pytest
import numpy as np
from api.lib.optimizers import *
from api.lib.preprocessing.initializers import random_normal, ones


class TestGradientDescent:

    def test_compute_gradients(self, session):
        trainable = (ones(2, 3), ones(1, 3))

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
    def test_minimize(self, session, lr, mocker):
        trainable = (ones(2, 3), ones(1, 3))

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

        minimize_ops = optimizer.minimize(op)
        session.run(minimize_ops)

        w_expected = w_init_value - np.ones(W.value.shape) * lr
        assert np.array_equal(W.value, w_expected)

        b_expected = b_init_value - np.ones(b.value.shape) * lr
        assert np.array_equal(b.value, b_expected)
