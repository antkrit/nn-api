"""Contains useful frequently used objects."""
import numpy as np
from api.lib.autograd.namespace import nodes as nodes_container


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
        raise ValueError(f"Cannot match sizes: {len(data)} and {len(placeholders)}")

    return {
        p.name: iter(np.atleast_2d(data[i]))
        for i, p in enumerate(placeholders)
    }


def convert_to_tensor(type_, *args, **kwargs):
    """Create node with given parameters.

    :param type_: str, type of the node to create
    :return: created node
    """
    return nodes_container[type_](*args, **kwargs)
