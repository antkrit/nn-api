import numpy as np


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
        p.name: (x for x in np.atleast_2d(data[i]))
        for i, p in enumerate(placeholders)
    }
