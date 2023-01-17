"""Contains implementation of Gradient Descent optimizers (GD, SGD)"""
from api.lib.optimization.base import Optimizer, _MinimizeOp


class GradientDescent(Optimizer):
    r"""Gradient Descent optimizer.

    The most basic and classic optimization algorithm
    which is commonly-used in neural networks.

    Update rule: :math:`x_{k+1} = x_k - \alpha \nabla f'`

    :param lr: some small positive value
    :type learning_rate: float
    """

    def __init__(self, lr):
        """Constructor method
        """
        # Attention: C0103 disabled(invalid-name)
        # because lr is much better than learning_rate, prove me wrong
        # pylint: disable=C0103
        self.learning_rate = lr

    def minimize(self, floss):
        def _update_rule(grad):
            return -self.learning_rate*grad

        return _MinimizeOp(floss, _update_rule)
