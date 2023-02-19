"""Contains classes and functions to work with graph."""
# Attention: W0611 disabled(unused-import)
# because pylint doesn't recognize objects in code samples for doctest
# pylint: disable=W0611
from api.lib.autograd.node import (
    Variable, Placeholder, Operation, topological_sort, AssignOperation
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
        """Forward propagation aver a graph.

        Computes the output of a target operation using one sample.

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

        :param target: node or list of nodes to perform the forward step for
        :param feed_dict: data for placeholders
        :raises KeyError: in case where there are no values in feed_dict for
            the empty Placeholder
        :return: value of the last node, i.e. result of graph
        """
        feed_dict = feed_dict or {}
        outputs = []

        try:
            while True:
                for sorted_ in topological_sort(target):
                    print(sorted_)
                    for node in sorted_:
                        if isinstance(node, Placeholder):
                            node.value = next(feed_dict[node.name])
                        if isinstance(node, Operation):

                            inputs = [x.value for x in node.inputs]
                            node.value = node.forward(*inputs)

                    outputs.append(sorted_[-1].value)

                if not feed_dict:
                    return outputs[0] if len(outputs) == 1 else outputs

        except StopIteration:
            return outputs[0] if len(outputs) == 1 else outputs

    def gradients(self, target):
        """Compute gradients for the given graph.

        Function performs topological sort for the target node, then sets
        its gradient to zero and recursively computes all other gradients.

        >>> w = Variable(1, name='w')
        >>> x = Variable(2, name='x')
        >>> op = w * x
        >>> session = Session()
        >>> # d(op)/dw = x, d(op)/dx = w, d(op)/d(op) = 1
        >>> session.gradients(op)
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
            >>> session.gradients(op)
            {w: 2.0, x: 1.0, graph-0/operator-multiply-6: 1.0}

        :param target: head nodes of the graph
        :return: a dict with node and its gradient pairs
        """
        order = next(topological_sort(target))

        visited = set()
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
