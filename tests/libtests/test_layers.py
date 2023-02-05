import pytest
import numpy as np
import api.lib.autograd as ag
import api.lib.activation as actv
import api.lib.preprocessing.initializers as init
from api.lib.layers import *


@pytest.mark.parametrize(
    'x',
    [1, [1, 1, 1]],
    ids=['scalar', 'vector']
)
@pytest.mark.parametrize(
    'p_activation',
    [actv.Swish, actv.Sigmoid, actv.Tanh],
    ids=['_activation=swish', '_activation=sigmoid', '_activation=tanh']
)
@pytest.mark.parametrize(
    'p_weight_init',
    [init.random_normal, init.random_uniform],
    ids=['_w_init=random_n', '_w_init=random_u']
)
def test_dense_layer(session, x, p_activation, p_weight_init):
    p_actv = p_activation(session=session)

    test_case = np.atleast_2d(x)
    tc_size = (test_case.size, test_case.size)
    x = ag.Variable(test_case)

    layer = Dense(
        size=tc_size,
        activation=p_actv,
        weight_initializer=p_weight_init
    )

    assert np.array_equal(session.run(layer(x)), session.run(layer.forward(x)))
    # TODO: gradient of the layer test
