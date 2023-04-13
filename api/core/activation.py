"""Contains implementation of commonly used activation functions."""
import abc

import numpy as np

from api.core import autograd as ag

__all__ = ("Sigmoid", "Tanh", "Swish", "Softmax", "Softplus", "ReLU", "ELU")


class BaseActivation:
    """Base activation class.

    Calling an activation function returns an operation by
    default, not a value.

    To create a custom layer, it is enough to override the
    ``__init__()`` and ``forward()`` methods. ``forward()`` should
    be used to implement activation logic, an algorithm that
    will calculate the "activated" output.

    So, a simple activation can be implemented as this:

    .. code-block:: python

        class Linear(BaseActivation):

            def forward(self, x):
                return x

    :param threshold: some minute value to avoid problems like
        div by 0 or log(0), defaults to 0
    :param session: current session, if None - creates new, defaults to None
    """

    def __init__(self, session=None, threshold=0):
        self.session = session or ag.Session()
        self.threshold = threshold

    @abc.abstractmethod
    def forward(self, x):
        """Calculate function."""
        raise NotImplementedError("Must be implemented in subclasses.")

    def __call__(self, x, *args, **kwargs):
        return ag.utils.node_wrapper(self.forward, x, *args, **kwargs)


class Sigmoid(BaseActivation):
    r"""Sigmoid activation function.

    The output of a sigmoid function ranges between 0 and 1. Mostly used for
    models where we have to predict the "probability".

    Activation rule: :math:`\dfrac{1}{1 + e^{-x}}`
    """

    def forward(self, x):
        """Calculate sigmoid."""
        return 1 / (1 + ag.ops.exp(-x))


class Tanh(BaseActivation):
    r"""Hyperbolic Tangent activation function.

    This function is relatively similar to :class:`Sigmoid`, but with one big
    advantage - function is 0-centric(output is in range (-1, 1)).

    Activation rule: :math:`\dfrac{e^{x} - e^{-x}}{e^{x} + e^{-x}}`
    """

    def forward(self, x):
        """Calculate tanh."""
        return (2 / (1 + ag.ops.exp(-2 * x))) - 1


class ReLU(BaseActivation):
    r"""(Leaky) Rectified Linear Unit activation function.

    Solve gradient saturation problem, because output range is (-inf, inf).
    If the :math:`\alpha` parameter is 0, there is "dead ReLU" problem where the
    function is completely inactive with negative input values.

    Activation rule: :math:`max(\alpha x, x)`

    :param alpha: leaky coefficient, defaults to 0
    """

    def __init__(self, alpha=0, session=None, threshold=0):
        """Constructor method."""
        super().__init__(session, threshold)
        self.alpha = alpha

    def forward(self, x):
        """Calculate (l)relu."""
        return ag.ops.max(x, self.alpha * x)


class ELU(BaseActivation):
    r"""Exponential Linear Unit activation function.

    Fixes some :class:`ReLU` issues, allows to learn faster.

    :param alpha: leaky coefficient, defaults to 1
    """

    def __init__(self, alpha=1, session=None, threshold=0):
        """Constructor method."""
        super().__init__(session, threshold)
        self.alpha = alpha

    def forward(self, x):
        """Calculate elu."""
        cond = np.asarray(self.session.run(x)) > 0

        cond_true = ag.utils.convert_to_node(value=np.asarray(cond, dtype=int))
        cond_false = ag.utils.convert_to_node(
            value=np.asarray(np.invert(cond), dtype=int)
        )
        return cond_true * x + cond_false * self.alpha * (ag.ops.exp(x) - 1)


class Softmax(BaseActivation):
    r"""Softmax activation function.

    Mostly used for multi-class classification problems.

    The activation rule is equivalent to:

    .. code-block:: python

        exp / (ag.ops.sum(exp) + self.threshold)
    """

    def forward(self, x):
        """Calculate softmax."""
        shiftx = x - np.max(self.session.run(x))
        exp = ag.ops.exp(shiftx)
        return exp / (ag.ops.sum(exp) + self.threshold)


class Swish(BaseActivation):
    r"""Self-Gated activation function.

    Swish is unbounded above and bounded below. Unlike :class:`ReLU`,
    the function is smooth and non-monotonic.

    Activation rule: :math:`x*sigmoid(\beta x)`

    :param beta: either constant or trainable parameter, defaults to 1
    """

    def __init__(self, beta=1, session=None, threshold=1e-32):
        """Constructor method."""
        super().__init__(session, threshold)
        self.beta = beta

    def forward(self, x):
        """Calculate self-gated function."""
        return x / (1 + ag.ops.exp(-self.beta * x) + self.threshold)


class Softplus(BaseActivation):
    """Softplus activation function.

    Relatively smoother version of the :class:`ReLU`.

    Activation rule: :math:`ln(1 + e^x)`
    """

    def forward(self, x):
        """Calculate softplus."""
        return ag.ops.log(1 + ag.ops.exp(x) + self.threshold)
