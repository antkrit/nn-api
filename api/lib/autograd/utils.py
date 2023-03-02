"""Contains useful frequently used objects."""
import numpy as np
from api.lib.utils import Container
from api.lib.autograd.node import (
    Node, Variable, Placeholder, Constant,
    Operation, UnaryOperation, BinaryOperation
)


NODES_CONTAINER = Container(
    name='nodes',

    node=Node,
    constant=Constant,
    variable=Variable,
    placeholder=Placeholder,
    operation=Operation,
    unary_operation=UnaryOperation,
    binary_operation=BinaryOperation
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

    return {
        p.name: iter(np.atleast_2d(data[i]))
        for i, p in enumerate(placeholders)
    }


def convert_to_tensor(type_, *args, **kwargs):
    """Create node with given parameters.

    :param type_: str, type of the node to create
    :return: created node
    """
    return NODES_CONTAINER[type_](*args, **kwargs)


def fill_placeholders(*placeholders, feed_dict):
    """Fill placeholders with value from feed_dict without running session."""
    for pl in placeholders:
        if isinstance(pl, Placeholder):
            pl.value = feed_dict.get(pl.name, None)
