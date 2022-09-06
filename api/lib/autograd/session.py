"""Contains classes and functions to work with graph."""
from collections import deque
from api.lib.autograd.graph import Variable, Placeholder, Operation


class Session:
    """A class for running graph operations.

    Basically graph is a parsed mathematical expression
    consisting of a list of nodes. To get the result of such
    expression, it is necessary to create a Session and call the
    `run` method.

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

        If there are placeholders in the graph you need to fill them
        with data. Pass that data to feed_dict in node_name:data format.

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

    .. note::
        If there are placeholders, it is necessary run forward propagation
        first or manually fill them with data. Otherwise, an error will be raised.

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


def topological_sort(head_node):
    """Perform topological sort for a given graph using DFS algorithm.

    :param head_node: node to start sorting from
    :param head_node: :class:`Node`
    :return: list  of sorted nodes
    :rtype: deque
    """
    visited = set()
    order = deque()

    def _dfs(node):
        """Depth-first search recursion helper."""
        nonlocal visited

        if node not in visited:
            visited.add(node)
            if isinstance(node, Operation):
                for input_node in node.inputs:
                    _dfs(input_node)

            order.append(node)

    _dfs(head_node)
    return order
