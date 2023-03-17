"""Contains implementation of commonly used activation functions."""
import abc
import numpy as np
from api.core import autograd as ag


__all__ = ('Sigmoid', 'Tanh', 'Swish', 'Softmax', 'Softplus', 'ReLU', 'ELU')


class BaseActivation:
    """Base activation class.

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

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in subclasses.")


class Sigmoid(BaseActivation):
    """Sigmoid activation function."""

    def __init__(self, session=None, threshold=0):
        """Constructor method."""
        super().__init__(session, threshold)

    def forward(self, x):
        """Calculate sigmoid.

        The output of a sigmoid function ranges between 0 and 1. Mostly used for
        models where we have to predict the "probability".
        """
        return 1 / (1 + ag.ops.exp(-x))

    def __call__(self, x, *args, **kwargs):
        return ag.utils.node_wrapper(self.forward, x, *args, **kwargs)


class Tanh(BaseActivation):
    """Hyperbolic Tangent activation function."""

    def __init__(self, session=None, threshold=0):
        """Constructor method."""
        super().__init__(session, threshold)

    def forward(self, x):
        """Calculate tanh.

        This function is relatively similar to class:`Sigmoid`, but with one big
        advantage - function is 0-centric(output is in range (-1, 1)).
        """
        return (2 / (1+ag.ops.exp(-2*x))) - 1

    def __call__(self, x, *args, **kwargs):
        return ag.utils.node_wrapper(self.forward, x, *args, **kwargs)


class ReLU(BaseActivation):
    """(Leaky) Rectified Linear Unit activation function.

    :param alpha: leaky coefficient, defaults to 0
    """

    def __init__(self, alpha=0, session=None, threshold=0):
        """Constructor method."""
        super().__init__(session, threshold)
        self.alpha = alpha

    def forward(self, x):
        """Calculate (l)relu.

        Solve gradient saturation problem, because output range is (-inf, inf).
        If the `a` parameter is 0, there is "dead ReLU" problem where the
        function is completely inactive with negative input values.
        """
        return ag.ops.max(x, self.alpha*x)

    def __call__(self, x, *args, **kwargs):
        return ag.utils.node_wrapper(self.forward, x, *args, **kwargs)


class ELU(BaseActivation):
    """Exponential Linear Unit activation function.

    :param alpha: leaky coefficient, defaults to 1
    :param session: current session
    """

    def __init__(self, alpha=1, session=None, threshold=0):
        """Constructor method."""
        super().__init__(session, threshold)
        self.alpha = alpha

    def forward(self, x):
        """Calculate elu.

        Fixes some class:`ReLU` issues, allows learn faster.

        :param x: input value
        """
        cond = np.asarray(self.session.run(x)) > 0

        cond_true = ag.utils.convert_to_node(value=np.asarray(cond, dtype=int))
        cond_false = ag.utils.convert_to_node(
            value=np.asarray(np.invert(cond), dtype=int)
        )
        return cond_true*x + cond_false*self.alpha*(ag.ops.exp(x)-1)

    def __call__(self, x, *args, **kwargs):
        return ag.utils.node_wrapper(self.forward, x, *args, **kwargs)


class Softmax(BaseActivation):
    """Softmax activation function."""

    def __init__(self, session=None, threshold=1e-9):
        """Constructor method."""
        super().__init__(session, threshold)

    def forward(self, x):
        """Calculate softmax.

        Mostly used for multi-class classification problems.

        :param x: input value
        """
        shiftx = x - np.max(self.session.run(x))
        e = ag.ops.exp(shiftx)
        return e / (ag.ops.sum(e)+self.threshold)

    def __call__(self, x, *args, **kwargs):
        return ag.utils.node_wrapper(self.forward, x, *args, **kwargs)


class Swish(BaseActivation):
    """Self-Gated activation function.

    :param beta: either constant or trainable parameter, defaults to 1
    """

    def __init__(self, beta=1, session=None, threshold=1e-32):
        """Constructor method."""
        super().__init__(session, threshold)
        self.beta = beta

    def forward(self, x):
        """Calculate self-gated function.

        Swish is unbounded above and bounded below. Unlike class:`ReLU`,
        the function is smooth and non-monotonic.

        :param x: input value
        """
        return x / (1 + ag.ops.exp(-self.beta*x)+self.threshold)

    def __call__(self, x, *args, **kwargs):
        return ag.utils.node_wrapper(self.forward, x, *args, **kwargs)


class Softplus(BaseActivation):
    """Softplus activation function."""

    def __init__(self, session=None, threshold=1e-32):
        """Constructor method."""
        super().__init__(session, threshold)

    def forward(self, x):
        """Calculate softplus.

        Relatively smoother version of the class:`ReLU`.

        :param x: input value
        """
        return ag.ops.log(1+ag.ops.exp(x)+self.threshold)

    def __call__(self, x, *args, **kwargs):
        return ag.utils.node_wrapper(self.forward, x, *args, **kwargs)
