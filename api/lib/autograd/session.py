"""Contains classes and functions to work with graph."""
# Attention: W0611 disabled(unused-import)
# because pylint doesn't recognize objects in code samples for doctest
# pylint: disable=W0611
from api.lib.autograd.node import (
    Variable, Placeholder, Operation, topological_sort
)


class Session:
    """A class for running graph operations.

    Basically graph is a parsed mathematical expression
    consisting of a list of nodes. To get the result of such
    expression, it is necessary to create a Session and call the
    `run` method.

    .. note::
        Until the step forward is taken, head node or some expression should be
        treated as an execution strategy to get the value of that node. It
        follows that class:`Operation` nodes have no own value, only operands.

    :Example:

    >>> a = Variable(10)
    >>> b = Variable(20)
    >>> c = a + b  # Node.__add__(a, b)
    >>> session = Session()
    >>> session.run(c)
    30.0
    """
    def run(self, target, feed_dict=None):
        """Forward propagation aver a graph. Computes the output of
         a target operation.

        .. note::
            If there are placeholders in the graph you need to fill them
            with data. Otherwise, KeyError will be raised. Pass that data
            to feed_dict in node_name:data format.

            >>> a = 2
            >>> b = Placeholder('x')
            >>> c = a * b  # Node.__mul__(a, b)
            >>> session = Session()
            >>> session.run(c, feed_dict={'x':15})
            30.0

        :param target: last node of the graph
        :type target: :class:`Node`
        :param feed_dict: data for placeholders
        :param feed_dict: dict, optional
        :raises KeyError: in case where there are no values in feed_dict for \
        the empty Placeholder
        :return: value of the last node, i.e. result of graph
        :rtype: np.array
        """
        feed_dict = feed_dict or {}
        sorted_ = topological_sort(target)

        for node in sorted_:
            if isinstance(node, Placeholder) and node.value is None:
                node.value = feed_dict[node.name]
            if isinstance(node, Operation):
                node.value = node.forward(*[x.value for x in node.inputs])

        return sorted_[-1].value


def gradients(target):
    """Get gradient of the loss w.r.t. the node's value.

        >>> w = Variable(1, name='w')
        >>> x = Variable(2, name='x')
        >>> op = w * x
        >>> gradients(op)  # d(op)/dw = x, d(op)/dx = w, d(op)/d(op) = 1
        {w: 2.0, x: 1.0, graph-0/operator-multiply-5: 1.0}

    .. warning::
        To calculate the gradient, it is important to run forward step first
        so that all class:`Operation` nodes have their own value.

    .. note::
        If there are placeholders, it is necessary run forward
        propagation first or manually fill them with data.
        Otherwise, TypeError error will be raised.

        >>> w = Variable(1, name='w')
        >>> x = Placeholder(name='x')
        >>> op = w * x
        >>> session = Session()
        >>> _ = session.run(op, feed_dict={'x': 2.0})
        >>> gradients(op)
        {w: 2.0, x: 1.0, graph-0/operator-multiply-6: 1.0}

    :param target: target node
    :type target: class:`Node`
    :return: a dict with node and its gradient pairs
    :rtype: dict
    """
    visited = set()

    order = topological_sort(target)
    order[-1].gradient = 1.0  # df/df

    for node in reversed(order):
        if isinstance(node, Operation):
            inputs = node.inputs
            grads = node.backward(
                *[x.value for x in inputs],
                dout=node.gradient
            )

            for inp, grad in zip(inputs, grads):
                if inp in visited:
                    inp.gradient += grad
                else:
                    inp.gradient = grad
                visited.add(inp)

    return {node: node.gradient for node in order}
