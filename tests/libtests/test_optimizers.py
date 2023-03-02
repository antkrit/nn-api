import pytest
import numpy as np
from api.lib.optimizers import *
from api.lib.preprocessing.initializers import ones
from api.lib import namespace


@pytest.mark.parametrize('lr', [0.01, 2], ids=['lr=0.01', 'lr=2'])
def test_gradient_descent(session, lr):
    trainable = (ones(2, 3), ones(1, 3))

    optimizer = GradientDescent(
        learning_rate=lr,
        trainable_variables=trainable,
        session=session
    )

    W, b = trainable
    w_init_value, b_init_value = W.value.copy(), b.value.copy()
    noise = np.random.randint(0, 10)
    X = ones(*W.value.shape[::-1]) + noise

    op = X @ W + b

    minimize_op = optimizer.minimize(op)
    assert isinstance(minimize_op, namespace.nodes.operation)

    session.run([op, minimize_op])

    w_expected = w_init_value - (np.ones(W.value.shape) + noise) * lr
    assert np.array_equal(W.value, w_expected)

    b_expected = b_init_value - np.ones(b.value.shape) * lr
    assert np.array_equal(b.value, b_expected)
