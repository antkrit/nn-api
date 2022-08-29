"""Contains base optimizer class and Minimize Operation."""
from abc import ABC, abstractmethod
from api.lib.autograd import gradients, Operation, Variable


class Optimizer(ABC):
    """Base optimizer class."""

    @abstractmethod
    def minimize(self, *args, **kwargs):
        """Init and return `_MinimizeOp`."""
        raise NotImplementedError("Must be implemented in child classes.")


class _MinimizeOp(Operation):
    """Minimization operation.

    This operation must be returned by all optimizers using
    the "minimize" method. This operation is responsible for
    updating values of graph variables, i.e. weights and biases.
    In particular, this operation is responsible for finding gradients.
    The variable update rule is set dynamically, depending on the optimizer.

    :param floss: head node of the graph
    :type floss: class:`Node`
    :param frule: variable update rule
    :type frule: function
    :param options: additional options for the update rule
    :type options: dict, optional
    """

    def __init__(self, floss, frule, options=None):
        """Constructor method
        """
        super().__init__('minimize')

        self.options = options or {}
        self.floss = floss
        self.frule = frule

    def forward(self):
        """Forward pass."""
        grads = gradients(self.floss)
        for node in grads:
            if isinstance(node, Variable):
                grad = grads[node]
                self.options.update({'grad': grad})
                node.value += self.frule(**self.options)

    def backward(self, *args, **kwargs):
        raise NotImplementedError(
            "There is no gradient for the minimization operation."
        )
