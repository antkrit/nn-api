"""Contains implementation of Gradient Descent optimizers (GD, SGD)"""
from api.lib.bases import BaseOptimizer


class GradientDescent(BaseOptimizer):
    r"""Gradient Descent optimizer.

    The most basic and classic optimization algorithm
    which is commonly-used in neural networks.

    Update rule: :math:`x_{k+1} = x_k - \alpha \nabla f'`

    :param lr: some small positive value, learning rate
    """

    DEFAULT_TRAINABLE = ('weight', 'bias')

    def __init__(self, lr, trainable_variables=DEFAULT_TRAINABLE):
        """Constructor method."""
        # Attention: C0103 disabled(invalid-name)
        # because lr is much better than learning_rate, prove me wrong
        # pylint: disable=C0103
        super().__init__(trainable_variables)
        self.learning_rate = lr

    def compute_gradient(self, *args, **kwargs):
        """Compute gradients for trainable variables."""
        pass

    def apply_gradient(self, *args, **kwargs):
        """Apply computed gradients to trainable variables."""
        pass

    def minimize(self, floss, *args, **kwargs):
        pass
