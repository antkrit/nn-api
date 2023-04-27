import numpy as np
import pytest

from api.core import autograd as ag
from api.core import namespace
from api.core.optimizers import *
from api.core.preprocessing.initializers import ones

np.seterr(all="ignore")


def test_base_optimizer(session, mocker):
    with pytest.raises(TypeError):
        _ = BaseOptimizer(trainable_variables=[])

    mocker.patch.multiple(BaseOptimizer, __abstractmethods__=set())

    test_var = namespace.nodes.variable(value=1, name="x")
    base = BaseOptimizer(trainable_variables=[test_var])

    with pytest.raises(namespace.exceptions.NoGradient):
        base.backward()

    var_ref = base.add_variable(init_val=1, var_name="added")
    assert isinstance(var_ref, namespace.nodes.variable)
    assert np.allclose(var_ref.value, 1)
    assert var_ref in base.variables()

    var = base.add_variable_from_reference(var_ref, var_name="added+ref")
    assert isinstance(var, namespace.nodes.variable)
    assert np.allclose(var.value, 0)
    assert var in base.variables()

    var = base.add_variable_from_reference(
        var_ref, var_name="added+ref", init=ones
    )
    var_shape = var.value.shape if hasattr(var.value, "shape") else ()
    assert isinstance(var, namespace.nodes.variable)
    assert np.allclose(var.value, np.ones(var_shape))
    assert var in base.variables()

    var_shape = (2, 2)
    var = base.add_variable_from_reference(
        var_ref, var_name="added+ref", init=ones, shape=var_shape
    )
    assert isinstance(var, namespace.nodes.variable)
    assert np.allclose(var.value, np.ones(var_shape))
    assert var in base.variables()

    op = (test_var**2) / 2  # d(op)/dx = x
    assert not base.inputs

    base.minimize(op)
    assert base.inputs and isinstance(base.inputs[0], namespace.nodes.constant)
    assert base.inputs[0].value is op

    mocker.patch.object(
        BaseOptimizer,
        "apply_gradient",
        return_value=namespace.nodes.variable(1),
    )

    session.run(op)

    if hasattr(base, "_iteration") and base._iteration is None:
        base._iteration = namespace.nodes.variable(0)

    assert base.forward(op) == 1
    assert session.run(base._iteration) == 1

    gradients = [-1234, 1, 1234]
    assert np.array_equal(base.clip(gradients), gradients)

    base = BaseOptimizer(clipvalue=10, trainable_variables=[])
    assert np.array_equal(base.clip(gradients), np.clip(gradients, -10, 10))

    base = BaseOptimizer(clipnorm=2, trainable_variables=[])
    assert np.linalg.norm(base.clip(gradients)) == 3


def test_base_optimizer_docstring_example(session):
    class SimpleGD(BaseOptimizer):
        def __init__(self, learning_rate=0.1, trainable=None, name="gd"):
            super().__init__(trainable, session=None, name=name)
            self.learning_rate = learning_rate

            self._lr = None
            self._built = False

        def build(self, var_list):
            super().build(var_list)
            if self._built:
                return

            self._lr = self.add_variable(self.learning_rate, "learning_rate")
            self._built = True

        def apply_gradient(self, x, grad):
            return ag.ops.assign_add(x, -self._lr * grad)

    X = ones((3, 3))
    x_init_value = X.value.copy()
    lr = 0.01

    objective = (X**2) / 2  # d(op)/dx = x

    optimizer = SimpleGD(
        learning_rate=lr,
        trainable=[X],
    )

    minimize_op = optimizer.minimize(objective)
    session.run(objective, minimize_op)

    x_expected = x_init_value - x_init_value * lr
    assert np.allclose(X.value, x_expected)


@pytest.mark.parametrize("lr", [0.01, 2], ids=["lr=0.01", "lr=2"])
def test_gradient_descent(session, lr):
    X = ones((3, 3))
    x_init_value = X.value.copy()

    objective = (X**2) / 2  # d(op)/dx = x

    with pytest.raises(ValueError):
        _ = GradientDescent(
            learning_rate=lr,
            momentum=999,
            trainable_variables=[X],
            session=session,
        )

    with pytest.raises(ValueError):
        _ = GradientDescent(
            learning_rate=lr,
            momentum=-999,
            trainable_variables=[X],
            session=session,
        )

    optimizer = GradientDescent(
        learning_rate=lr, momentum=0, trainable_variables=[X], session=session
    )

    minimize_op = optimizer.minimize(objective)
    session.run(objective, minimize_op)

    x_expected = x_init_value - x_init_value * lr
    assert np.allclose(X.value, x_expected)

    session.run(minimize_op)

    x_expected = x_expected - x_expected * lr
    assert np.allclose(X.value, x_expected)


@pytest.mark.parametrize("lr", [0.01, 2], ids=["lr=0.01", "lr=2"])
def test_adagrad(session, lr):
    X = ones((3, 3))
    x_init_value = X.value.copy()

    objective = (X**2) / 2  # d(op)/dx = x

    optimizer = Adagrad(
        learning_rate=lr,
        initial_accumulator_value=0,
        trainable_variables=[X],
        session=session,
    )

    minimize_op = optimizer.minimize(objective)
    session.run(objective, minimize_op)

    accumulator = np.zeros(X.value.shape) + x_init_value**2
    acc_grad = x_init_value / np.sqrt(accumulator + optimizer.threshold)
    x_expected = x_init_value - lr * acc_grad
    assert np.allclose(X.value, x_expected)


@pytest.mark.parametrize("lr", [0.01, 2], ids=["lr=0.01", "lr=2"])
def test_adadelta(session, lr):
    X = ones((3, 3))
    x_init_value = X.value.copy()

    objective = (X**2) / 2  # d(op)/dx = x

    rho = 0
    optimizer = Adadelta(
        learning_rate=lr, rho=rho, trainable_variables=[X], session=session
    )

    minimize_op = optimizer.minimize(objective)
    session.run(objective, minimize_op)

    rms = lambda val: np.sqrt(val + optimizer.threshold)

    accumulated_grad = np.zeros(x_init_value.shape)
    accumulated_delta_var = np.zeros(x_init_value.shape)

    accumulated_grad = (
        rho * accumulated_grad + (1 - rho) * x_init_value * x_init_value
    )
    delta = -rms(accumulated_delta_var) * x_init_value / rms(accumulated_grad)
    x_expected = x_init_value + lr * delta
    assert np.allclose(X.value, x_expected)


@pytest.mark.parametrize("lr", [0.01, 2], ids=["lr=0.01", "lr=2"])
def test_rmsprop(session, lr):
    X = ones((3, 3))
    x_init_value = X.value.copy()

    objective = (X**2) / 2  # d(op)/dx = x

    rho = 0
    optimizer = RMSProp(
        learning_rate=lr, rho=rho, trainable_variables=[X], session=session
    )

    minimize_op = optimizer.minimize(objective)
    session.run(objective, minimize_op)

    velocity = np.zeros(x_init_value.shape)

    velocity = rho * velocity + (1 - rho) * x_init_value**2
    denominator = velocity + optimizer.threshold

    increment = lr * x_init_value * (1 / np.sqrt(denominator))
    x_expected = x_init_value - increment

    assert np.allclose(X.value, x_expected)


@pytest.mark.parametrize(
    "amsgrad", [True, False], ids=["amsgrad=True", "amsgrad=False"]
)
@pytest.mark.parametrize("lr", [0.01, 2], ids=["lr=0.01", "lr=2"])
def test_adam(session, lr, amsgrad):
    X = ones((3, 3))
    x_init_value = X.value.copy()

    objective = (X**2) / 2  # d(op)/dx = x

    beta_1 = 0.9
    beta_2 = 0.999
    beta_1_power = beta_1**1
    beta_2_power = beta_2**1

    optimizer = Adam(
        learning_rate=lr,
        beta_1=beta_1,
        beta_2=beta_2,
        amsgrad=amsgrad,
        trainable_variables=[X],
        session=session,
    )

    minimize_op = optimizer.minimize(objective)
    session.run(objective, minimize_op)

    m = np.zeros(x_init_value.shape)
    v = np.zeros(x_init_value.shape)

    alpha = (
        lr
        * np.sqrt(1 - beta_2_power)
        / (1 - beta_1_power + optimizer.threshold)
    )

    m = (x_init_value - m) * (1 - beta_1)
    v = v + (((x_init_value**2) - v) * (1 - beta_2))

    if optimizer.amsgrad:
        v_hat = np.zeros(x_init_value.shape)
        v = np.maximum(v_hat, v)

    x_expected = x_init_value - (m * alpha) / (np.sqrt(v) + optimizer.threshold)
    assert np.allclose(X.value, x_expected)


@pytest.mark.parametrize("lr", [0.01, 2], ids=["lr=0.01", "lr=2"])
def test_adamax(session, lr):
    X = ones((3, 3))
    x_init_value = X.value.copy()

    objective = (X**2) / 2  # d(op)/dx = x

    beta_1 = 0.9
    beta_2 = 0.999
    beta_1_power = beta_1**1

    optimizer = Adamax(
        learning_rate=lr,
        beta_1=beta_1,
        beta_2=beta_2,
        trainable_variables=[X],
        session=session,
    )

    minimize_op = optimizer.minimize(objective)
    session.run(objective, minimize_op)

    m = np.zeros(x_init_value.shape)
    u = np.zeros(x_init_value.shape)

    m = m + (x_init_value - m) * (1 - beta_1)
    u = np.maximum(beta_2 * u, np.abs(x_init_value))
    x_expected = x_init_value - (lr * m) / (
        (1 - beta_1_power) * (u + optimizer.threshold)
    )
    assert np.allclose(X.value, x_expected)
