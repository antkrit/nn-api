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
    def __init__(self):
        """Constructor method."""
        self.__CTX_GLOBAL_TOKEN = 'globals'
        self.__context = {self.__CTX_GLOBAL_TOKEN: []}

    def run(self, target, feed_dict=None):
        """Forward propagation aver a graph.

        Computes the output of a target operation using one sample.

        .. note::
            If there are placeholders in the graph you need to fill them
            with data. Otherwise, TypeError will be raised. Pass that data
            to feed_dict in node_name:data format.

            >>> a = 2
            >>> b = Placeholder('x')
            >>> c = a * b  # Node.__mul__(a, b)
            >>> session = Session()
            >>> session.run(c, feed_dict={'x':15})
            30.0

        :param target: node or list of nodes to perform the forward step for
        :param feed_dict: data for placeholders
        :return: value of the last node, i.e. result of graph
        """
        feed_dict = feed_dict or {}
        session_output_token = 'FORWARD_OUTPUT'
        output = {}

        try:
            # infinite cycle is used to make sure all values
            # from the feed dict entry will be used (iterate until there
            # are no values in the batch).
            while True:
                for sorted_ in topological_sort(target):
                    for node in sorted_:
                        self.__process_node_forward(node, feed_dict=feed_dict)

                    head_node = sorted_[-1]
                    head_node_entry = output.get(head_node, None)

                    try:
                        # add a new node output to the list
                        head_node_entry.append(head_node.value)
                    except AttributeError:
                        if head_node_entry is not None:
                            # if there is single node output in the
                            # output dict, create list with the new one
                            output[head_node] = [
                                output[head_node],
                                head_node.value
                            ]
                        else:
                            # if this node's output does not exist in the
                            # output dict, set it to a single value
                            output[head_node] = head_node.value

                if not feed_dict:
                    # If there is no feed dict, then there is no batch.
                    break

        except StopIteration:
            # If there are no more values from the feed dict
            # entry (which is supposed to be an iterator) - StopIteration error
            # will be raised (for implementation see the
            # __process_node_forward -> __handle_placeholder_forward function)
            pass
        finally:
            self.ctx_add(session_output_token, output)
            results = list(output.values())
            return results[0] if len(results) == 1 else results

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

    def ctx_add(self, v_name, v_value):
        """Add a variable to the session context.

        :param v_name: variable name
        :param v_value: value of the variable
        :return: None
        """
        if v_name not in self.__context:
            self.ctx_add(self.__CTX_GLOBAL_TOKEN, v_name)
        self.__context.setdefault(v_name, []).append(v_value)

    def ctx_get(self, v_name):
        """Get a variable from the session context.

        :param v_name: variable name
        :return: value of the variable or None if there is no such value
        """
        return self.__context.get(v_name, None)

    @staticmethod
    def __handle_operation_forward(node):
        """Handle operation node.

        :param node: operation node to process
        """
        inputs = [x.value for x in node.inputs]
        node.value = node.forward(*inputs)

    @staticmethod
    def __handle_placeholder_forward(node, feed_dict):
        """Handle placeholder node.

        :param node: placeholder node to process
        :param feed_dict: original feed_dict
        :raises StopIteration: if the no more data for the placeholder entry
        """
        feed_dict = feed_dict or {}

        try:
            pl_iter = feed_dict[node.name]
            node.value = next(pl_iter)
        except KeyError:
            node.value = node.value

    def __process_node_forward(self, node, feed_dict=None):
        """Get a node as input and process it depending on the class.

        :param node: node to process
        :param feed_dict: original feed dict (used to handle placeholders),
            defaults to None
        :return: handled node (variable nodes are returned in
            their original form)
        """
        if isinstance(node, Placeholder):
            self.__handle_placeholder_forward(node, feed_dict)
        elif isinstance(node, Operation):
            self.__handle_operation_forward(node)

        return node
