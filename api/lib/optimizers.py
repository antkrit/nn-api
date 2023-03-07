"""Contains implementation of Gradient Descent optimizers (GD, SGD)"""
from api.lib.bases import BaseOptimizer
from api.lib.autograd import ops
from api.lib.preprocessing.initializers import NormalInitializer


__all__ = (
    'GradientDescent', 'Adagrad', 'Adadelta', 'RMSProp', 'Adam',
    'Adamax',
)


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
    :param trainable_variables: nodes, variables to minimize
    :param session: current session, defaults to None
    """

    def __init__(
            self,
            learning_rate=0.001,
            momentum=0,
            nesterov=False,
            trainable_variables=None,
            session=None,
            name='GradientDescent'
    ):
        """Constructor method."""
        super().__init__(trainable_variables, session, name=name)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov

        # should be initialized with build() method
        self._lr = None
        self._momentum = None
        self._momentums = None

        self.__built = False

        if isinstance(momentum, (int, float)) and (
            momentum < 0 or momentum > 1
        ):
            raise ValueError("momentum must be in range [0, 1].")

    def build(self, var_list):
        """Initialize optimizer variables."""
        super().build(var_list)
        if self.__built:
            return

        self._lr = self.add_variable(self.learning_rate, 'learning_rate')
        self._momentum = self.add_variable(self.momentum, 'momentum')

        self._momentums = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    model_var=var, var_name="m"
                )
            )

        self.__built = True

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
            else:
                return ops.assign_add(x, m)
        else:
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
    :param trainable_variables: nodes, variables to minimize
    :param session: current session, defaults to None
    """

    def __init__(
            self,
            learning_rate=0.1,
            initial_accumulator_value=0.1,
            epsilon=1e-16,
            trainable_variables=None,
            session=None,
            name='Adagrad'
    ):
        """Constructor method."""
        super().__init__(
            trainable_variables,
            session,
            name=name,
            threshold=epsilon
        )
        self.learning_rate = learning_rate
        self.initial_accumulator_value = initial_accumulator_value

        # should be initialized with build() method
        self._lr = None
        self._accumulators = None

        self.__built = False

    def build(self, var_list):
        """Initialize optimizer variables."""
        super().build(var_list)
        if self.__built:
            return

        self._lr = self.add_variable(self.learning_rate, 'learning_rate')

        self._accumulators = []
        for var in var_list:
            self._accumulators.append(
                self.add_variable_from_reference(
                    model_var=var,
                    var_name='accumulator',
                    init=NormalInitializer(
                        mu=self.initial_accumulator_value,
                        sigma=0
                    )
                )
            )

        self.__built = True

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
    :param trainable_variables: nodes, variables to minimize
    :param session: current session, defaults to None
    """

    def __init__(
            self,
            learning_rate=1,
            rho=0.9,
            epsilon=1e-16,
            trainable_variables=None,
            session=None,
            name='Adadelta'
    ):
        """Constructor method."""
        super().__init__(
            trainable_variables,
            session,
            name=name,
            threshold=epsilon
        )
        self.learning_rate = learning_rate
        self.rho = rho

        # should be initialized with build() method
        self._lr = None
        self._rho = None
        self._accumulated_grads = None
        self._accumulated_delta_vars = None

        self.__built = False

    def build(self, var_list):
        """Initialize optimizer variables."""
        super().build(var_list)
        if self.__built:
            return

        self._lr = self.add_variable(self.learning_rate, 'learning_rate')
        self._rho = self.add_variable(self.rho, 'rho')

        self._accumulated_grads = []
        self._accumulated_delta_vars = []
        for var in var_list:
            self._accumulated_grads.append(
                self.add_variable_from_reference(var, 'accumulated_grad')
            )
            self._accumulated_delta_vars.append(
                self.add_variable_from_reference(var, 'accumulated_delta_var')
            )

        self.__built = True

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
            accumulated_grad,
            rho * accumulated_grad + (1 - rho) * grad * grad
        )
        delta_var = -rms(accumulated_delta_var) * grad / rms(accumulated_grad)
        accumulated_delta_var = ops.assign(
            accumulated_delta_var,
            rho * accumulated_delta_var + (1 - rho) * delta_var * delta_var
        )

        # required to run an assignment operation for accumulated_delta_var
        zero_value = 0*accumulated_delta_var

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
    :param trainable_variables: nodes, variables to minimize
    :param session: current session, defaults to None
    """

    def __init__(
            self,
            learning_rate=0.1,
            momentum=0,
            rho=0.95,
            epsilon=1e-16,
            centered=False,
            trainable_variables=None,
            session=None,
            name='RMSProp'
    ):
        """Constructor method."""
        super().__init__(
            trainable_variables,
            session,
            name=name,
            threshold=epsilon
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

        self.__built = False

    def build(self, var_list):
        """Initialize optimizer variables."""
        super().build(var_list)
        if self.__built:
            return

        self._lr = self.add_variable(self.learning_rate, 'learning_rate')
        self._momentum = self.add_variable(self.momentum, 'momentum')
        self._rho = self.add_variable(self.rho, 'rho')

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

        self.__built = True

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
            velocity,
            rho * velocity + (1 - rho) * ops.pow(grad, 2)
        )
        if self.centered:
            average_grad = ops.assign(
                average_grad,
                rho * average_grad + (1 - rho) * grad
            )
            denominator = velocity - ops.pow(average_grad, 2) + self.threshold
        else:
            denominator = velocity + self.threshold

        increment = lr * grad * ops.rsqrt(denominator)
        if self.momentum:
            momentum = ops.assign(
                momentum,
                self.momentum * momentum + increment
            )
            return ops.assign_add(x, -momentum)
        else:
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
    :param trainable_variables: nodes, variables to minimize
    :param session: current session, defaults to None
    """

    def __init__(
            self,
            learning_rate=0.1,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-16,
            amsgrad=False,
            trainable_variables=None,
            session=None,
            name='Adam'
    ):
        """Constructor method."""
        super().__init__(
            trainable_variables,
            session,
            name=name,
            threshold=epsilon
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

        self.__built = False

    def build(self, var_list):
        """Initialize optimizer variables."""
        super().build(var_list)
        if self.__built:
            return

        self._lr = self.add_variable(self.learning_rate, 'learning_rate')
        self._beta_1 = self.add_variable(self.beta_1, 'beta_1')
        self._beta_2 = self.add_variable(self.beta_2, 'beta_2')

        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    model_var=var, var_name="m"
                )
            )
            self._velocities.append(
                self.add_variable_from_reference(
                    model_var=var, var_name="v"
                )
            )
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(
                        model_var=var, var_name="vhat"
                    )
                )

        self.__built = True

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

        alpha = lr * ops.sqrt(1 - beta_2_power) / (1 - beta_1_power + self.threshold)

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
        moment estimates, defaults to 0.9.
    :param epsilon: small floating point value used to maintain numerical
        stability, defaults to 1e-16
    :param trainable_variables: nodes, variables to minimize
    :param session: current session, defaults to None
    """

    def __init__(
            self,
            learning_rate=0.1,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-16,
            trainable_variables=None,
            session=None,
            name='Adamax'
    ):
        """Constructor method."""
        super().__init__(
            trainable_variables,
            session,
            name=name,
            threshold=epsilon
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

        self.__built = False

    def build(self, var_list):
        """Initialize optimizer variables."""
        super().build(var_list)
        if self.__built:
            return

        self._lr = self.add_variable(self.learning_rate, 'learning_rate')
        self._beta_1 = self.add_variable(self.beta_1, 'beta_1')
        self._beta_2 = self.add_variable(self.beta_2, 'beta_2')

        self._momentums = []
        self._norm = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    model_var=var, var_name="m"
                )
            )
            self._norm.append(
                self.add_variable_from_reference(
                    model_var=var, var_name="u"
                )
            )

        self.__built = True

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
            x,
            -(lr * m) / ((1 - beta_1_power) * (u + self.threshold))
        )
