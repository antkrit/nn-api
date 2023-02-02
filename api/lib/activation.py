"""Contains implementation of commonly used activation functions."""
import numpy as np
import api.lib.autograd as ag
from api.lib.bases import BaseActivation


__all__ = ('Sigmoid', 'Tanh', 'Swish', 'Softmax', 'Softplus', 'ReLU', 'ELU')


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
        return 1 / (1 + ag.exp(-x))

    def __call__(self, x, *args, **kwargs):
        return ag.node_wrapper(self.forward, x, *args, **kwargs)


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
        return (2 / (1+ag.exp(-2*x))) - 1

    def __call__(self, x, *args, **kwargs):
        return ag.node_wrapper(self.forward, x, *args, **kwargs)


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
        return ag.max(x, self.alpha*x)

    def __call__(self, x, *args, **kwargs):
        return ag.node_wrapper(self.forward, x, *args, **kwargs)


class ELU(BaseActivation):
    """Exponential Linear Unit activation function.

    :param alpha: leaky coefficient, defaults to 1
    """

    def __init__(self, alpha=1, session=None, threshold=0):
        """Constructor method."""
        super().__init__(session, threshold)
        self.alpha = alpha

    def forward(self, x):
        """Calculate elu.

        Fixes some class:`ReLU` issues, allows learn faster.

        :param x: input value
        :param alpha: leaky coefficient, defaults to 1
        :param session: current session
        """
        cond = self.session.run(x) > 0

        cond_true = ag.Constant(np.asarray(cond, dtype=int))
        cond_false = ag.Constant(np.asarray(np.invert(cond), dtype=int))
        return cond_true*x + cond_false*self.alpha*(ag.exp(x)-1)

    def __call__(self, x, *args, **kwargs):
        return ag.node_wrapper(self.forward, x, *args, **kwargs)


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
        e = ag.exp(shiftx)
        return e / (ag.sum(e)+self.threshold)

    def __call__(self, x, *args, **kwargs):
        return ag.node_wrapper(self.forward, x, *args, **kwargs)


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
        return x / (1 + ag.exp(-self.beta*x)+self.threshold)

    def __call__(self, x, *args, **kwargs):
        return ag.node_wrapper(self.forward, x, *args, **kwargs)


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
        return ag.log(1+ag.exp(x)+self.threshold)

    def __call__(self, x, *args, **kwargs):
        return ag.node_wrapper(self.forward, x, *args, **kwargs)
