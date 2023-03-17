import pytest
import numpy as np
import api.lib.activation as actv
import api.lib.preprocessing.initializers as init
from api.lib.layers import *
from api.lib.autograd import Variable, Placeholder


def test_base_layer(mocker, test_case_unary):
    with pytest.raises(TypeError):
        _ = BaseLayer()

    mocker.patch.multiple(BaseLayer, __abstractmethods__=set())

    bl = BaseLayer()
    assert not bl._built

    bl.build(input_shape=())
    assert bl._built

    size = (1, 1)
    var = bl.add_variable(shape=size, initializer=init.ones, trainable=True)
    assert np.array_equal(var.value, np.ones(size)) and var.shape == size
    assert var in bl.variables()

    mocker.patch.object(
        BaseLayer,
        'forward',
        side_effect=lambda x, *args, **kwargs: x
    )

    size = (1, 1)
    layer_output = bl(None, input_shape=size)
    assert isinstance(layer_output, Placeholder)
    assert layer_output.shape == size
    assert bl.input is layer_output
    assert bl.input_shape == layer_output.shape


def test_base_layer_docstring_example(session, test_case_unary):
    class Linear(BaseLayer):

        def __init__(self, units=10, name='Linear'):
            super().__init__(session=None, name=name)
            self.units = units

            self.weights = None
            self.bias = None

        def build(self, input_shape):
            self.weights = initializers.random_normal(
                size=[input_shape[-1], self.units]
            )
            self.bias = initializers.ones(
                size=[1, self.units]
            )
            self._built = True

        def forward(self, value, *args, **kwargs):
            return value @ self.weights + self.bias

    lnr = Linear(units=10)
    output = lnr(test_case_unary)
    expected = np.array(test_case_unary).dot(lnr.weights.value) + lnr.bias.value
    assert np.array_equal(session.run(output), expected)


@pytest.mark.parametrize(
    'x',
    [1, [1, 1, 1]],
    ids=['scalar', 'vector']
)
@pytest.mark.parametrize(
    'activation',
    ['swish', actv.ReLU(alpha=0.01), 'tanh', None],
    ids=[
        '_activation=swish',
        '_activation=relu(callable)',
        '_activation=tanh',
        '_activation=None'
    ]
)
@pytest.mark.parametrize(
    'weight_init',
    ['random_normal', init.random_uniform, None],
    ids=['_w_init=random_n', '_w_init=random_u(callable)', '_w_init=None']
)
@pytest.mark.parametrize(
    'use_bias',
    [True, False],
    ids=['with bias', 'without bias']
)
def test_dense_layer(session, x, activation, weight_init, use_bias):
    test_case = np.atleast_2d(x)
    units = np.random.randint(1, 5)
    x = Variable(test_case)

    layer = Dense(
        units=units,
        activation=activation,
        weight_initializer=weight_init,
        bias_initializer=weight_init,
        use_bias=use_bias
    )

    layer_output = session.run(layer(x))
    assert np.array_equal(layer_output, session.run(layer.forward(x)))

    expected_output_shape = (test_case.shape[-2], layer.units)
    assert np.array_equal(layer_output.shape, expected_output_shape)


@pytest.mark.parametrize('shape', [(1,), (1, 1)], ids=['shape-1d', 'shape-2d'])
def test_input_layer(session, shape):
    with pytest.raises(ValueError):
        _ = Input(input_shape=())

    inp = Input(input_shape=shape)
    output = inp()
    assert isinstance(output, Placeholder)
    assert output.shape == shape

    inp = Input(input_shape=shape)
    with pytest.raises(ValueError):
        different_shape = (2, 3)
        _ = inp(x=np.ones(different_shape))

    assert inp(x=np.ones(shape)) is not None
