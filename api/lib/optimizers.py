"""Contains implementation of Gradient Descent optimizers (GD, SGD)"""
from operator import itemgetter
from api.lib.bases import BaseOptimizer
from api.lib.autograd import ops
from api.lib.autograd.utils import convert_to_tensor


__all__ = ('GradientDescent', )


class GradientDescent(BaseOptimizer):
    r"""Gradient Descent optimizer.

    The most basic and classic optimization algorithm
    which is commonly-used in neural networks.

    Update rule: :math:`x_{k+1} = x_k - \alpha \nabla f'`

    :param lr: some small positive value, learning rate
    :param trainable_variables: nodes, variables to minimize
    :param session: current session, defaults to None
    """

    def __init__(self, lr, trainable_variables=None, session=None):
        """Constructor method."""
        super().__init__(trainable_variables, session)
        self._lr = convert_to_tensor(
            'variable',
            value=lr,
            name='learning_rate'
        )

    def compute_gradients(self, out):
        """Compute gradients for trainable variables.

        :param out: output of the network
        :return: gradient of the loss w.r.t trainable vars, all gradients
        """
        grd = self.session.gradients(out)
        trainable_grd = itemgetter(*self.trainable)(grd)

        return trainable_grd, grd

    def apply_gradient(self, x, grad):
        """Apply computed gradients to trainable variables using update rule.

        :param x: node, value to update
        :param grad: gradient of the loss w.r.t x
        :return: updated node
        """
        return ops.assign_add(x, -self._lr * grad)

    def minimize(self, floss, *args, **kwargs):
        """Compute gradients and apply an update rule to trainable variables.

        :parma floss: output of the network, head node
        :return: list of operations
        """
        trainable_gradients, _ = self.compute_gradients(floss)
        return [
            self.apply_gradient(x, grd)
            for x, grd in zip(self.trainable, trainable_gradients)
        ]
