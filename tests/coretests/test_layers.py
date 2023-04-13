import numpy as np
import pytest

from api.core import activation as actv
from api.core.autograd import Placeholder, Variable
from api.core.layers import *
from api.core.preprocessing import initializers as init


def test_base_layer(mocker, test_case_unary):
    with pytest.raises(TypeError):
        _ = BaseLayer()

    mocker.patch.multiple(BaseLayer, __abstractmethods__=set())

    bl = BaseLayer()
    assert not bl.built

    with pytest.raises(ValueError):
        bl.built = True

    bl.build(input_shape=())
    assert bl._built

    size = (1, 1)
    var = bl.add_variable(shape=size, initializer=init.ones, trainable=True)
    assert np.array_equal(var.value, np.ones(size)) and var.shape == size
    assert var in bl.variables()

    mocker.patch.object(
        BaseLayer, "forward", side_effect=lambda x, *args, **kwargs: x
    )

    size = (1, 1)
    layer_output = bl(None, input_shape=size)
    assert isinstance(layer_output, Placeholder)
    assert layer_output.shape == size
    assert bl.input is layer_output


def test_base_layer_docstring_example(session, test_case_unary):
    class Linear(BaseLayer):
        def __init__(self, units=10, name="Linear"):
            super().__init__(session=None, name=name)
            self.units = units

            self.weights = None
            self.bias = None

        def build(self, input_shape):
            self.weights = init.random_normal(
                size=[input_shape[-1], self.units]
            )
            self.bias = init.ones(size=[1, self.units])
            self._built = True

        def forward(self, value, *args, **kwargs):
            return value @ self.weights + self.bias

    lnr = Linear(units=10)
    output = lnr(test_case_unary)
    expected = np.array(test_case_unary).dot(lnr.weights.value) + lnr.bias.value
    assert np.array_equal(session.run(output), expected)


@pytest.mark.parametrize(
    "x",
    [np.ones(()), np.ones((3,)), np.ones((3, 3)), np.ones((4, 3, 3))],
    ids=["scalar", "vector", "matrix", "tensor"],
)
@pytest.mark.parametrize(
    "activation",
    ["swish", actv.ReLU(alpha=0.01), "tanh", None],
    ids=[
        "_activation=swish",
        "_activation=relu(callable)",
        "_activation=tanh",
        "_activation=None",
    ],
)
@pytest.mark.parametrize(
    "weight_init",
    ["random_normal", init.random_uniform, None],
    ids=["_w_init=random_n", "_w_init=random_u(callable)", "_w_init=None"],
)
@pytest.mark.parametrize(
    "use_bias", [True, False], ids=["with bias", "without bias"]
)
def test_dense_layer(session, x, activation, weight_init, use_bias):
    batch_test_case = np.atleast_3d(x)
    units = np.random.randint(1, 5)
    x = Variable(batch_test_case)

    layer = Dense(
        units=units,
        activation=activation,
        weight_initializer=weight_init,
        bias_initializer=weight_init,
        use_bias=use_bias,
    )

    layer_output = session.run(layer(x))
    assert np.array_equal(layer_output, session.run(layer.forward(x)))

    expected_output_shape = (*batch_test_case.shape[:-1], layer.units)
    assert layer_output.shape == expected_output_shape


@pytest.mark.parametrize("shape", [(1,), (1, 1)], ids=["shape-1d", "shape-2d"])
def test_input_layer(session, shape):
    with pytest.raises(ValueError):
        _ = Input(input_shape=())

    inp = Input(input_shape=shape)
    output = inp()
    assert isinstance(output, Placeholder)
    assert output.shape == shape

    inp = Input(input_shape=shape)
    out = inp(x=np.ones(shape))
    assert out is not None

    out_not_cached = inp(x=np.ones(shape))
    assert out_not_cached is not None
    assert out_not_cached is not out

    out = inp(x=None)
    assert out is not None

    out_is_cached = inp(x=None)
    assert out_is_cached is not None
    assert out_is_cached is out

    inp = Input(input_shape=shape, batch_size=1)
    assert inp.shape == shape
    assert inp.batch == 1
    assert inp.batch_shape == (1, *shape)


@pytest.mark.parametrize(
    "x",
    [np.ones(()), np.ones((3,)), np.ones((3, 3)), np.ones((4, 3, 3))],
    ids=["scalar", "vector", "matrix", "tensor"],
)
def test_multiple_layers(session, x):
    test_case = np.atleast_3d(x)
    hidden_units = np.random.randint(2, 5)

    inp = Input(input_shape=test_case.shape)()
    hl = Dense(units=hidden_units)
    out = Dense(units=1)

    op = out(hl(inp), input_shape=hl.batch_shape)

    inp.value = test_case

    layer_output = session.run(op, feed_dict={inp.name: np.zeros(inp.shape)})
    assert layer_output.shape[-1] == 1


def test_input_shape():
    shape = (1,)
    input_shape = InputShape(shape)
    assert input_shape.shape == shape
    assert input_shape.batch is None
    assert input_shape.batch_input_shape == (None, *shape)

    shape = (1, 1)
    input_shape = InputShape(shape)
    assert input_shape.shape == shape[1:]
    assert input_shape.batch == 1
    assert input_shape.batch_input_shape == shape

    shape = (1, 1, 1)
    input_shape = InputShape(shape)
    assert input_shape.shape == shape[1:]
    assert input_shape.batch == 1
    assert input_shape.batch_input_shape == shape
