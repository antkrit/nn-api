"""Contains implementation of optimizers."""
import abc

import numpy as np

from api.core.autograd import Constant, Operation, Session, Variable, ops
from api.core.exception import NoGradientException
from api.core.preprocessing.initializers import NormalInitializer

__all__ = (
    "BaseOptimizer",
    "GradientDescent",
    "Adagrad",
    "Adadelta",
    "RMSProp",
    "Adam",
    "Adamax",
)

# optimizers have many abbreviations (such as lr=learning_rate) well known
# to users. Therefore, for convenience, C0103 has been disabled in this file
# pylint: disable=invalid-name


class BaseOptimizer(Operation, metaclass=abc.ABCMeta):
    """Base optimizer class.

    Basically, every optimizer is an Operation. So, in most cases,
    to create a custom optimizer, it is enough to override the ``__init__()``,
    ``build()`` and ``apply_gradient()`` methods:

    - ``build()`` should be used to add new optimizer variables and will be
      called inside the ``self.forward()`` method (in general, this method was
      created to add some inner context to each optimizer, e.g. momentums).
      Each child implementation should call the parent's ``build()`` method
      at the beginning.
    - ``apply_gradient()`` should be used to implement optimizer logic, an
      algorithm that will be used to update a variable.

    .. warning::
        Any other methods are not recommended to be overridden.

    So, a simple optimizer can be implemented as this:

    .. code-block:: python

        class SimpleGD(BaseOptimizer):

            def __init__(self, learning_rate=0.1, trainable=None, name='gd'):
                super().__init__(trainable, session=None, name=name)
                self.learning_rate = learning_rate

                self._lr = None
                self._built = False

            def build(self, var_list):
                super().build(var_list)
                if self._built:
                    return

                self._lr = self.add_variable(
                    self.learning_rate, 'learning_rate'
                )
                self._built = True

            def apply_gradient(self, x, grad):
                return autograd.ops.assign_add(x, -self._lr * grad)

    :param trainable_variables: variables to optimize, defaults to None
    :param clipnorm: if set, the gradient of each weight is individually
        clipped so that its norm is no higher than this value.
    :param clipvalue: if set, the gradient of each weight is clipped to be
        no higher than this value.
    :param session: current session, if None - creates new, defaults to None
    :param name: optimizer name
    """

    def __init__(
        self,
        trainable_variables,
        clipnorm=0,
        clipvalue=0,
        session=None,
        name=None,
        **kwargs,
    ):
        """Constructor method."""
        super().__init__(name=name, **kwargs)
        self.session = session or Session()
        self.trainable = trainable_variables or []
        self._variables = []

        self.clipnorm = clipnorm
        self.clipvalue = clipvalue

        # should be initialized with build() method
        self._iteration = None
        self._built = False

        self._index_dict = {}

    @abc.abstractmethod
    def build(self, var_list):
        """Initialize optimizer variables.

        This method should be implemented and called by subclasses.
        """
        if hasattr(self, "_built") and self._built:
            return

        self._iteration = self.add_variable(0, self.name + "/local-iteration")
        self._build_indexed_dict(var_list)

    def _build_indexed_dict(self, var_list):
        """Assign index to each trainable node."""
        self._index_dict = {}

        for i, v in enumerate(var_list):
            self._index_dict[v] = i

        return self._index_dict

    def add_variable(self, init_val, var_name=None):
        """Add new optimizer value.

        :param init_val: value of a variable to be created
        :param var_name: name of the node to be created
        :returns: an optimizer variable
        """
        var = Variable(value=init_val, name=var_name)
        self._variables.append(var)

        return var

    def add_variable_from_reference(
        self, model_var, var_name, init=None, shape=None
    ):
        """Create an optimizer variable from model variable.

        :param model_var: :class:`Variable` node. The corresponding model
            variable to the optimizer variable to be created.
        :param var_name: new optimizer value will be created with name
            'model_var.name/var_name'
        :param init: the initial value or initializer(callable) of the
            optimizer variable, if None, the initial value will be
            default to 0, defaults to None
        :param shape: shape of the optimizer variable value
        :returns: an optimizer variable
        """
        model_var_numpy = np.asarray(model_var.value)

        if init is None:
            init = np.zeros(model_var_numpy.shape)
        elif callable(init):
            if shape is None:
                shape = model_var_numpy.shape

            # init supposed to be of BaseInitializer type,
            # which returns node
            init = init(shape).value

        model_var_name = model_var.name.split("/")
        var_name = f"{model_var_name[-1]}/{var_name}"

        var = Variable(value=init, name=var_name)
        self._variables.append(var)

        return var

    def variables(self):
        """Return all variables."""
        return self._variables

    def clip(self, gradients):
        """Clip gradients if clipnorm or clipvalue is set."""
        if self.clipvalue:
            return [
                np.clip(g, -self.clipvalue, self.clipvalue) for g in gradients
            ]

        if self.clipnorm:
            return [
                self.clipnorm * (g / (np.linalg.norm(g) + 1e-16))
                if np.linalg.norm(g) >= self.clipnorm
                else g
                for g in gradients
            ]

        # `Session.gradient()` returns either 1 value or array of values
        # see `BaseOptimizer.forward()` implementation
        gradients = np.asarray(gradients, dtype=object)
        return np.atleast_1d(gradients).tolist()

    @abc.abstractmethod
    def apply_gradient(self, *args, **kwargs):
        """Apply computed gradients to trainable variables."""
        raise NotImplementedError("Must be implemented in child classes.")

    def forward(self, objective):
        """Compute gradients and apply an update rule to trainable variables.

        :parma objective: objective function, for which the gradients
            will be calculated
        :return: list of results
        """
        self.build(self.trainable)

        gradients = self.session.gradients(objective, returns=self.trainable)
        clipped_gradients = self.clip(gradients)

        apply_ops = [
            self.apply_gradient(x, grad=Variable(clipped_gradients[i]))
            for i, x in enumerate(self.trainable)
        ]
        iteration = ops.assign_add(self._iteration, 1)

        return self.session.run(*apply_ops, iteration, returns=apply_ops)

    def backward(self, *args, **kwargs):
        """Return gradient of the operation by given inputs."""
        msg = f"There is no gradient for operation {self.name}."
        raise NoGradientException(msg)

    def minimize(self, operation):
        """Set operation [to minimize] for the optimizer."""
        name = self.name + "/target-wrapper"
        wrapped_operation = Constant(value=operation, name=name)
        self.inputs = (wrapped_operation,)

        return self


class GradientDescent(BaseOptimizer):
    r"""Gradient Descent optimizer.

    Update rule for parameter `w` with gradient `g` when `momentum` is 0:

    .. code-block:: python

        w = w - learning_rate * g

    Update rule when `momentum` is larger than 0:

    .. code-block:: python

        velocity = momentum * velocity - learning_rate * g
        w = w + velocity

    When `nesterov=True`, this rule becomes:

    .. code-block:: python

        velocity = momentum * velocity - learning_rate * g
        w = w + momentum * velocity - learning_rate * g


    :param learning_rate: hyperparameter, some small value, defaults to 0.001
    :param momentum: hyperparameter, value in range [0, 1], defaults to 0
    :param nesterov: whether to apply Nesterov momentum, defaults to False
    """

    def __init__(
        self,
        learning_rate=0.001,
        momentum=0,
        nesterov=False,
        clipvalue=0,
        clipnorm=0,
        trainable_variables=None,
        session=None,
        name="GradientDescent",
    ):
        """Constructor method."""
        super().__init__(
            trainable_variables=trainable_variables,
            clipvalue=clipvalue,
            clipnorm=clipnorm,
            session=session,
            name=name,
        )
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov

        # should be initialized with build() method
        self._lr = None
        self._momentum = None
        self._momentums = None

        self._built = False

        if isinstance(momentum, (int, float)) and (
            momentum < 0 or momentum > 1
        ):
            raise ValueError("momentum must be in range [0, 1].")

    def build(self, var_list):
        super().build(var_list)
        if self._built:
            return

        self._lr = self.add_variable(self.learning_rate, "learning_rate")
        self._momentum = self.add_variable(self.momentum, "momentum")

        self._momentums = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(model_var=var, var_name="m")
            )

        self._built = True

    def apply_gradient(self, x, grad):
        """Apply computed gradients to trainable variables using update rule.

        :param x: node, value to update
        :param grad: gradient of the x node
        :return: update operation
        """
        m = self._momentums[self._index_dict[x]]
        lr = self._lr
        momentum = self._momentum

        if momentum:
            m = ops.assign(m, m * momentum - lr * grad)
            if self.nesterov:
                return ops.assign_add(x, m * momentum - lr * grad)

            return ops.assign_add(x, m)

        # pylint: disable=invalid-unary-operand-type
        # learning rate is defined with `build()` method
        return ops.assign_add(x, -lr * grad)


class Adagrad(BaseOptimizer):
    """Optimizer that implements the Adaptive Gradient algorithm.

    Adagrad is an optimizer with parameter-specific learning rates,
    which are adapted relative to how frequently a parameter gets
    updated during training. The more updates a parameter receives,
    the smaller the updates.

    :param learning_rate: hyperparameter, some small value, defaults to 0.1
    :param initial_accumulator_value: hyperparameter, value in range [0, 1],
        defaults to 0
    :param epsilon: small floating point value used to maintain numerical
        stability, defaults to 1e-16
    """

    def __init__(
        self,
        learning_rate=0.1,
        initial_accumulator_value=0.1,
        epsilon=1e-16,
        clipvalue=0,
        clipnorm=0,
        trainable_variables=None,
        session=None,
        name="Adagrad",
    ):
        """Constructor method."""
        super().__init__(
            trainable_variables=trainable_variables,
            clipvalue=clipvalue,
            clipnorm=clipnorm,
            session=session,
            name=name,
            threshold=epsilon,
        )
        self.learning_rate = learning_rate
        self.initial_accumulator_value = initial_accumulator_value

        # should be initialized with build() method
        self._lr = None
        self._accumulators = None

        self._built = False

    def build(self, var_list):
        super().build(var_list)
        if self._built:
            return

        self._lr = self.add_variable(self.learning_rate, "learning_rate")

        self._accumulators = []
        for var in var_list:
            self._accumulators.append(
                self.add_variable_from_reference(
                    model_var=var,
                    var_name="accumulator",
                    init=NormalInitializer(
                        mu=self.initial_accumulator_value, sigma=0
                    ),
                )
            )

        self._built = True

    def apply_gradient(self, x, grad):
        """Apply computed gradients to trainable variables using update rule.

        :param x: node, value to update
        :param grad: gradient of the x node
        :return: update operation
        """
        lr = self._lr
        accumulator = self._accumulators[self._index_dict[x]]

        accumulator = ops.assign_add(accumulator, grad**2)
        acc_grad = ops.div(grad, ops.sqrt(accumulator + self.threshold))

        # pylint: disable=invalid-unary-operand-type
        # learning rate is defined with `build()` method
        return ops.assign_add(x, -lr * acc_grad)


class Adadelta(BaseOptimizer):
    """Optimizer that implements the Adadelta algorithm.

    Adadelta is a more robust extension of Adagrad that adapts learning rates
    based on a moving window of gradient updates, instead of accumulating all
    past gradients. This way, Adadelta continues learning even when many
    updates have been done. Compared to Adagrad, in the original version of
    Adadelta you don't have to set an initial learning rate. In this
    implementation learning_rate can be set

    :param learning_rate: hyperparameter, some small value, defaults to 1
    :param rho: hyperparameter, the decay rate, defaults to 0.9
    :param epsilon: small floating point value used to maintain numerical
        stability, defaults to 1e-16
    """

    def __init__(
        self,
        learning_rate=1,
        rho=0.9,
        epsilon=1e-16,
        clipvalue=0,
        clipnorm=0,
        trainable_variables=None,
        session=None,
        name="Adadelta",
    ):
        """Constructor method."""
        super().__init__(
            trainable_variables=trainable_variables,
            clipvalue=clipvalue,
            clipnorm=clipnorm,
            session=session,
            name=name,
            threshold=epsilon,
        )
        self.learning_rate = learning_rate
        self.rho = rho

        # should be initialized with build() method
        self._lr = None
        self._rho = None
        self._accumulated_grads = None
        self._accumulated_delta_vars = None

        self._built = False

    def build(self, var_list):
        super().build(var_list)
        if self._built:
            return

        self._lr = self.add_variable(self.learning_rate, "learning_rate")
        self._rho = self.add_variable(self.rho, "rho")

        self._accumulated_grads = []
        self._accumulated_delta_vars = []
        for var in var_list:
            self._accumulated_grads.append(
                self.add_variable_from_reference(var, "accumulated_grad")
            )
            self._accumulated_delta_vars.append(
                self.add_variable_from_reference(var, "accumulated_delta_var")
            )

        self._built = True

    def apply_gradient(self, x, grad):
        """Apply computed gradients to trainable variables using update rule.

        :param x: node, value to update
        :param grad: gradient of the x node
        :return: update operation
        """
        lr = self._lr
        rho = self._rho

        accumulated_grad = self._accumulated_grads[self._index_dict[x]]
        accumulated_delta_var = self._accumulated_delta_vars[
            self._index_dict[x]
        ]

        def rms(val):
            return ops.sqrt(val + self.threshold)

        accumulated_grad = ops.assign(
            accumulated_grad, rho * accumulated_grad + (1 - rho) * grad * grad
        )
        delta_var = -rms(accumulated_delta_var) * grad / rms(accumulated_grad)
        accumulated_delta_var = ops.assign(
            accumulated_delta_var,
            rho * accumulated_delta_var + (1 - rho) * delta_var * delta_var,
        )

        # required to run an assignment operation for accumulated_delta_var
        zero_value = 0 * accumulated_delta_var

        return ops.assign_add(x, lr * delta_var + zero_value)


class RMSProp(BaseOptimizer):
    """Optimizer that implements the Root Mean Squared Propagation algorithm.

    This implementation of RMSProp uses plain momentum, not Nesterov momentum.

    The centered version additionally maintains a moving average of the
    gradients, and uses that average to estimate the variance.

    :param learning_rate: hyperparameter, some small value, defaults to 0.1
    :param momentum: hyperparameter, value in range [0, 1], defaults to 0
    :param rho: hyperparameter, the decay rate, defaults to 0.9
    :param epsilon: small floating point value used to maintain numerical
        stability, defaults to 1e-16
    :param centered: if True, gradients are normalized by the estimated
        variance of the gradient; if False, by the uncentered second moment.
        Setting this to True may help with training, but is slightly more
        expensive in terms of computation and memory, defaults to False
    """

    def __init__(
        self,
        learning_rate=0.1,
        momentum=0,
        rho=0.95,
        epsilon=1e-16,
        centered=False,
        clipvalue=0,
        clipnorm=0,
        trainable_variables=None,
        session=None,
        name="RMSProp",
    ):
        """Constructor method."""
        super().__init__(
            trainable_variables=trainable_variables,
            clipvalue=clipvalue,
            clipnorm=clipnorm,
            session=session,
            name=name,
            threshold=epsilon,
        )
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.rho = rho
        self.centered = centered

        # should be initialized with build() method
        self._lr = None
        self._momentum = None
        self._rho = None
        self._velocities = None
        self._momentums = None
        self._average_gradients = None

        self._built = False

    def build(self, var_list):
        super().build(var_list)
        if self._built:
            return

        self._lr = self.add_variable(self.learning_rate, "learning_rate")
        self._momentum = self.add_variable(self.momentum, "momentum")
        self._rho = self.add_variable(self.rho, "rho")

        self._velocities = []
        for var in var_list:
            self._velocities.append(
                self.add_variable_from_reference(var, "velocity")
            )

        self._momentums = []
        if self.momentum > 0:
            for var in var_list:
                self._momentums.append(
                    self.add_variable_from_reference(var, "momentum")
                )

        self._average_gradients = []
        if self.centered:
            for var in var_list:
                self._average_gradients.append(
                    self.add_variable_from_reference(var, "average_gradient")
                )

        self._built = True

    def apply_gradient(self, x, grad):
        """Apply computed gradients to trainable variables using update rule.

        :param x: node, value to update
        :param grad: gradient of the x node
        :return: update operation
        """
        lr = self._lr
        rho = self._rho

        velocity = self._velocities[self._index_dict[x]]
        momentum = None
        if self.momentum > 0:
            momentum = self._momentums[self._index_dict[x]]
        average_grad = None
        if self.centered:
            average_grad = self._average_gradients[self._index_dict[x]]

        velocity = ops.assign(
            velocity, rho * velocity + (1 - rho) * ops.pow(grad, 2)
        )
        if self.centered:
            average_grad = ops.assign(
                average_grad, rho * average_grad + (1 - rho) * grad
            )
            denominator = velocity - ops.pow(average_grad, 2) + self.threshold
        else:
            denominator = velocity + self.threshold

        increment = lr * grad * ops.rsqrt(denominator)
        if self.momentum:
            momentum = ops.assign(
                momentum, self.momentum * momentum + increment
            )
            return ops.assign_add(x, -momentum)

        return ops.assign_add(x, -increment)


class Adam(BaseOptimizer):
    """Optimizer that implements the Adam algorithm.

    Adam optimization is a stochastic gradient descent method that is based on
    adaptive estimation of first-order and second-order moments.

    :param learning_rate: hyperparameter, some small value, defaults to 0.1
    :param beta_1: hyperparameter, the exponential decay rate for the 1st
        moment estimates, defaults to 0.9.
    :param beta_2: hyperparameter, the exponential decay rate for the 2nd
        moment estimates, defaults to 0.9.
    :param epsilon: small floating point value used to maintain numerical
        stability, defaults to 1e-16
    :param amsgrad: whether to apply AMSGrad variant of this algorithm from
        the paper "On the Convergence of Adam and beyond", defaults to False
    """

    def __init__(
        self,
        learning_rate=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-16,
        amsgrad=False,
        clipvalue=0,
        clipnorm=0,
        trainable_variables=None,
        session=None,
        name="Adam",
    ):
        """Constructor method."""
        super().__init__(
            trainable_variables=trainable_variables,
            clipvalue=clipvalue,
            clipnorm=clipnorm,
            session=session,
            name=name,
            threshold=epsilon,
        )
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.amsgrad = amsgrad

        # should be initialized with build() method
        self._lr = None
        self._beta_1 = None
        self._beta_2 = None
        self._velocities = None
        self._momentums = None
        self._velocity_hats = None

        self._built = False

    def build(self, var_list):
        super().build(var_list)
        if self._built:
            return

        self._lr = self.add_variable(self.learning_rate, "learning_rate")
        self._beta_1 = self.add_variable(self.beta_1, "beta_1")
        self._beta_2 = self.add_variable(self.beta_2, "beta_2")

        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(model_var=var, var_name="m")
            )
            self._velocities.append(
                self.add_variable_from_reference(model_var=var, var_name="v")
            )
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(
                        model_var=var, var_name="vhat"
                    )
                )

        self._built = True

    def apply_gradient(self, x, grad):
        """Apply computed gradients to trainable variables using update rule.

        :param x: node, value to update
        :param grad: gradient of the x node
        :return: update operation
        """
        lr = self._lr
        beta_1 = self._beta_1
        beta_2 = self._beta_2
        beta_1_power = ops.pow(beta_1, self._iteration + 1)
        beta_2_power = ops.pow(beta_2, self._iteration + 1)

        m = self._momentums[self._index_dict[x]]
        v = self._velocities[self._index_dict[x]]

        alpha = lr * ops.sqrt(1 - beta_2_power)
        alpha /= 1 - beta_1_power + self.threshold

        m = ops.assign_add(m, (grad - m) * (1 - beta_1))
        v = ops.assign_add(v, (ops.pow(grad, 2) - v) * (1 - beta_2))
        if self.amsgrad:
            v_hat = self._velocity_hats[self._index_dict[x]]
            v = ops.assign(v_hat, ops.max(v_hat, v))

        return ops.assign_add(x, -(m * alpha) / (ops.sqrt(v) + self.threshold))


class Adamax(BaseOptimizer):
    """Optimizer that implements the Adamax algorithm.

    Adamax, a variant of Adam based on the infinity norm, is a first-order
    gradient-based optimization method

    :param learning_rate: hyperparameter, some small value, defaults to 0.1
    :param beta_1: hyperparameter, the exponential decay rate for the 1st
        moment estimates, defaults to 0.9.
    :param beta_2: hyperparameter, the exponential decay rate for the 2nd
        moment estimates, defaults to 0.999.
    :param epsilon: small floating point value used to maintain numerical
        stability, defaults to 1e-16
    """

    def __init__(
        self,
        learning_rate=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-16,
        clipvalue=0,
        clipnorm=0,
        trainable_variables=None,
        session=None,
        name="Adamax",
    ):
        """Constructor method."""
        super().__init__(
            trainable_variables=trainable_variables,
            clipvalue=clipvalue,
            clipnorm=clipnorm,
            session=session,
            name=name,
            threshold=epsilon,
        )
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        # should be initialized with build() method
        self._lr = None
        self._beta_1 = None
        self._beta_2 = None
        self._momentums = None
        self._norm = None

        self._built = False

    def build(self, var_list):
        super().build(var_list)
        if self._built:
            return

        self._lr = self.add_variable(self.learning_rate, "learning_rate")
        self._beta_1 = self.add_variable(self.beta_1, "beta_1")
        self._beta_2 = self.add_variable(self.beta_2, "beta_2")

        self._momentums = []
        self._norm = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(model_var=var, var_name="m")
            )
            self._norm.append(
                self.add_variable_from_reference(model_var=var, var_name="u")
            )

        self._built = True

    def apply_gradient(self, x, grad):
        """Apply computed gradients to trainable variables using update rule.

        :param x: node, value to update
        :param grad: gradient of the x node
        :return: update operation
        """
        lr = self._lr
        beta_1 = self._beta_1
        beta_2 = self._beta_2
        beta_1_power = ops.pow(beta_1, self._iteration + 1)

        m = self._momentums[self._index_dict[x]]
        u = self._norm[self._index_dict[x]]

        m = ops.assign_add(m, (grad - m) * (1 - beta_1))
        u = ops.assign(u, ops.max(beta_2 * u, ops.abs(grad)))
        return ops.assign_add(
            x, -(lr * m) / ((1 - beta_1_power) * (u + self.threshold))
        )
