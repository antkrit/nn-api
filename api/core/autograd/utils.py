"""Contains useful frequently used objects."""
from api.core.autograd.node import (
    Constant,
    Node,
    Operation,
    Placeholder,
    Variable,
)


def form_feed_dict(data, *placeholders):
    """Generate suitable feed dict for session.

    Each feed dict should contain pairs placeholder_name:iterable,
    where iterable is batch of data for each placeholder.

    :param data: list of data for each placeholder
    :param placeholders: placeholders to fill
    :raises ValueError: if not enough or too much data.
    :return: feed dict
    """
    if len(data) != len(placeholders):
        raise ValueError(
            f"Cannot match sizes: {len(data)} and {len(placeholders)}"
        )

    return {p.name: data[i] for i, p in enumerate(placeholders)}


def convert_to_node(
    value=None,
    to_constant=False,
    to_variable=False,
    to_placeholder=False,
    **kwargs,
):
    """Create node based on given input.

    If the value is of Node instance, return this node,
    If the value is None, create placeholder. In any other case,
    a variable will be created.

    To make sure that the desired type will be created,
    you need to set the corresponding argument to True.
    These arguments take precedence during creation.

    .. note::
        Operation node cannot be created in this way.

    .. note::
        If `to_placeholder` argument is True, a placeholder
        with the given value will be created.

    :param value: value of the Node to be created
    :param to_constant: create Constant node
    :param to_variable: create Variable node
    :param to_placeholder: create Placeholder node
    :return: created node
    """
    if isinstance(value, Node):
        return value

    if to_constant:
        return Constant(value, **kwargs)
    if to_variable:
        return Variable(value, **kwargs)
    if to_placeholder:
        plc = Placeholder(**kwargs)
        plc.value = value
        return plc

    if value is not None:
        return Variable(value, **kwargs)

    return Placeholder(**kwargs)


def fill_placeholders(*placeholders, feed_dict):
    """Fill placeholders with value from feed_dict without running session."""
    for plc in placeholders:
        if isinstance(plc, Placeholder):
            plc.value = feed_dict.get(plc.name, None)


def topological_sort(nodes):
    """Generates topological sort for a given graph using DFS algorithm.

    :param nodes: node to start sorting from
    :return: list  of sorted nodes
    """
    visited = set()
    order = []

    def _dfs(node):
        """Depth-first search recursion helper."""
        nonlocal visited, order

        if node not in visited:
            visited.add(node)
            if isinstance(node, Operation):
                for input_node in node.inputs:
                    _dfs(input_node)

            order.append(node)

    try:
        for node in iter(nodes):
            _dfs(node)
            yield order
            order, visited = [], set()
    except TypeError:
        _dfs(nodes)
        yield order


def node_wrapper(node, *args, **kwargs):
    """Automatically convert non-Node types to `Constant`.

    :raises TypeError: in case some operands are not Node
    """
    fnargs = []
    for arg in args:
        # in this implementation of the wrapper,
        # only numeric types are automatically converted
        # to a Constant node
        if isinstance(arg, Node):
            fnargs.append(arg)
        else:
            try:
                fnargs.append(Constant(arg))
            except TypeError as exc:
                raise TypeError(
                    f"Incompatible argument type: {type(arg)}."
                ) from exc

    return node(*fnargs, **kwargs)
