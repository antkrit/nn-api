"""Contains implementation of Gradient Descent optimizers (GD, SGD)"""
from api.lib.bases import BaseOptimizer
from api.lib.autograd import ops, utils


__all__ = ('GradientDescent', )


class GradientDescent(BaseOptimizer):
    r"""Gradient Descent optimizer.

    The most basic and classic optimization algorithm
    which is commonly-used in neural networks.

    Update rule: :math:`x_{k+1} = x_k - \alpha \nabla f'`

    :param learning_rate: some small positive value, learning rate
    :param trainable_variables: nodes, variables to minimize
    :param session: current session, defaults to None
    """

    def __init__(
            self,
            learning_rate,
            trainable_variables=None,
            session=None,
            name='GradientDescent'
    ):
        """Constructor method."""
        super().__init__(trainable_variables, session, name=name)
        self._lr = utils.convert_to_tensor(
            'variable',
            value=learning_rate,
            name='learning_rate'
        )

    def apply_gradient(self, x, grad):
        """Apply computed gradients to trainable variables using update rule.

        :param x: node, value to update
        :param grad: gradient of the loss w.r.t x
        :return: update operation
        """
        return ops.assign_add(x, -self._lr * grad)
