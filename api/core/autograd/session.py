"""Contains classes and functions to work with graph."""
# Attention: W0611 disabled(unused-import)
# because pylint doesn't recognize objects in code samples for doctest
# pylint: disable=W0611
import operator
from api.core.autograd.node import Variable, Placeholder, Operation
from api.core.autograd.utils import topological_sort


class Session:
    """A class for running graph operations.

    Basically graph is a parsed mathematical expression
    consisting of a list of nodes. To get the result of such
    expression, it is necessary to create a Session and call the
    :func:`run` method.

    .. note::
        Until the step forward is taken, Operation nodes have no own value,
        only operands. So, gradients should only be calculated after all the
        class:`Operation` nodes have their own value

    Every session has its own context. There is at least one field in each
    context: 'globals' consisting of a list of context variables (keys). Any
    other fields can be added manually using the :func:`ctx_add` method

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

    def run(self, *target, feed_dict=None, returns=None):
        """Forward propagation aver a graph.

        Computes the output of a target operation(s). If target is a list
        of operations then they will be performed in the specified order.

        .. warning::
           feed_dict values is supposed to be an iterator. Otherwise,
           TypeError will be raised. To avoid such errors - use
           :func:`form_feed_dict` utility.

        .. note::
            If there are placeholders in the graph you need to fill them
            with data. Otherwise, TypeError will be raised. Pass that data
            to the feed_dict as the node_name:data entry.

            >>> a = 2
            >>> b = Placeholder('x')
            >>> c = a * b  # Node.__mul__(a, b)
            >>> session = Session()
            >>> session.run(c, feed_dict={'x':iter([15])})
            30.0

        :param target: node or list of nodes to perform the forward step for
        :param feed_dict: data for placeholders
        :param returns: list of target resutls that need to be returned,
            if None - returns all targets resutls, defaults to None
        :return: value or list of value specified in the `returns` argument
        """
        feed_dict = feed_dict or {}
        run_output_token = 'FORWARD_OUTPUT'
        output = {}

        try:
            # infinite cycle is used to make sure all values
            # from the feed dict entry will be used (iterate until there
            # are no values in the batch).
            # note: If there are several feed_dict entries then the cycle
            # will work until no values remain in the largest of them.
            # Placeholders for which no data is left - will save the last value
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
                    # If there is no feed dict - stop iterate.
                    break

                if feed_dict and not self.ctx_get('placeholders'):
                    # if there is feed_dict and no placeholders - stop iterate
                    break

        except StopIteration:
            # If there are no more values from the feed dict
            # entry (which is supposed to be an iterator) - StopIteration error
            # will be raised (error raised by the
            # __process_node_forward -> __handle_placeholder_forward function)
            pass
        except KeyError:
            # no data for placeholder (error raised by the
            # __process_node_forward -> __handle_placeholder_forward function)
            raise
        except Exception:
            # TODO: log the error appropriately
            import logging
            import traceback
            logging.error(traceback.format_exc())

        if output:
            self.ctx_add(run_output_token, output)

            returns = returns or output.keys()
            return operator.itemgetter(*returns)(output)

    def gradients(self, target, returns=None):
        """Compute gradients for the given graph.

        Function performs topological sort for the target node, then sets
        its gradient to 1.0 and recursively computes all other gradients.

        >>> w = Variable(1, name='w')
        >>> x = Variable(2, name='x')
        >>> op = w * x
        >>> op.name = 'op'
        >>> session = Session()
        >>> # d(op)/dw = x, d(op)/dx = w, d(op)/d(op) = 1
        >>> session.gradients(op)
        {w: 2.0, x: 1.0, op: 1.0}

        .. warning::
            To calculate the gradient, it is important to run forward
            step first so that all class:`Operation` nodes have their
            own value. If there are placeholders, it is necessary run
            forward propagation first or manually fill them with data.
            Otherwise, TypeError error will be raised.

            >>> w = Variable(1, name='w')
            >>> x = Placeholder(name='x')
            >>> op = w * x
            >>> session = Session()
            >>> _ = session.run(op, feed_dict={'x': iter([2.0])})
            >>> session.gradients(op)
            {w: 2.0, x: 1.0, graph-0/operator-multiply-6: 1.0}

        :param target: head nodes of the graph
        :param returns: list of nodes whose gradient should be returned,
            if None - returns all targets resutls, defaults to None
        :return: a list of gradients
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

        grads = {node: node.gradient for node in order}
        self.ctx_add('gradients', grads)

        returns = returns or grads.keys()
        return operator.itemgetter(*returns)(grads)

    def ctx_add(self, v_name, v_value):
        """Add a variable to the session context.

        :param v_name: variable name
        :param v_value: value of the variable
        :return: None
        """
        if v_name not in self.__context:
            # automatically add unique names to globals.
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
        """Process operation node.

        :param node: operation node to process
        """
        inputs = [x.value for x in node.inputs]
        node.value = node.forward(*inputs)

    def __handle_placeholder_forward(self, node, feed_dict):
        """Process placeholder node.

        :param node: placeholder node to process
        :param feed_dict: original feed_dict
        :raises KeyError: if the placeholder is empty and there is no data
            for it in feed_dict
        :raises StopIteration: if the no more data for the placeholder entry
        """
        feed_dict = feed_dict or {}
        self.ctx_add('placeholders', node)

        try:
            pl_iter = feed_dict[node.name]
            node.value = next(pl_iter)
        except KeyError as exc:
            if node.value is not None:
                node.value = node.value
            else:
                raise KeyError(f"Unfilled placeholder '{node.name}'") from exc

    def __process_node_forward(self, node, feed_dict=None):
        """Get a node as input and process it depending on the class.

        :param node: node to process
        :param feed_dict: original feed dict (used to handle placeholders),
            defaults to None
        :return: processed node
        """
        if isinstance(node, Placeholder):
            self.__handle_placeholder_forward(node, feed_dict)
        elif isinstance(node, Operation):
            self.__handle_operation_forward(node)

        return node
