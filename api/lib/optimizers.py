"""Contains implementation of Gradient Descent optimizers (GD, SGD)"""
from api.lib.bases import BaseOptimizer


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

    def __init__(self, lr, trainable_variables, session=None):
        """Constructor method."""
        # Attention: C0103 disabled(invalid-name)
        # because lr is much better than learning_rate, prove me wrong
        # pylint: disable=C0103
        super().__init__(trainable_variables, session)
        self.lr = lr

    def compute_gradients(self, out):
        """Compute gradients for trainable variables.

        :param out: output of the network
        :return: gradient of the loss w.r.t trainable vars, all gradients
        """
        grd = self.session.gradients(out)
        trainable_grd = self._itemgetter(grd)
        return trainable_grd, grd

    def apply_gradient(self, x, grad):
        """Apply computed gradients to trainable variables using update rule.

        :param x: node, value to update
        :param grad: gradient of the loss w.r.t x
        :return: updated node
        """
        x.value -= self.lr * grad
        return x

    def minimize(self, floss, *args, **kwargs):
        """Compute gradients and apply an update rule to trainable variables.

        :parma floss: output of the network, head node
        :return: all computed gradients for the graph
        """
        train_grads, grd = self.compute_gradients(floss)
        for x, grd in zip(self.trainable, train_grads):
            self.apply_gradient(x, grd)

        return grd
