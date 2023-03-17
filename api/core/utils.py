"""Contains useful frequently used objects."""
import numpy as np
from collections.abc import MutableMapping


class Container(MutableMapping):
    """Base dict-like container class.

    To get an object use any of the three options
    >>> container = Container(name=..., obj=3)
    >>> container['obj']
    >>> container.obj
    >>> container('obj')

    To get the compiled instance - use __call__ method
    >>> container = Container(name=..., obj=lambda x: x)
    >>> container('obj_name', compiled=True, x=3)
    3
    """

    def __init__(self, name, *args, **kwargs):
        """Constructor method."""
        self.name = name
        self.store = dict()
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        return self.store[key]

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __call__(self, obj_name, compiled=False, *args, **kwargs):
        obj = self.__getitem__(key=obj_name)
        if callable(obj) and compiled:
            return obj(*args, **kwargs)
        return obj

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __repr__(self):
        return f'Container-{self.name}({self.store.items()})'


def at_least3d(array):
    """View inputs as arrays with at least three dimensions.

    Unlike the numpy counterpart, this function has the following behaviour:
    a 1-D array of shape (N,) becomes a view of shape (1, N, 1),
    and a 2-D array of shape (M, N) becomes a view of shape (1, M, N).

    :param array: array-like sequence
    :return: an array with ndim >= 3
    """
    array = np.asarray(array)

    if len(array.shape) == 0:
        return array.reshape((1, 1, 1))
    elif len(array.shape) == 1:
        return array[np.newaxis, :, np.newaxis]
    elif len(array.shape) == 2:
        return array[np.newaxis, :, :]
    else:
        return array


def unpack_x_y(data):
    """Unpacks user-provided data.

    This utility works with the following data forms: x; (x,); (x, y)

    :param data: data to unpack
    :raises ValueError: unexpected data format provided
    :return: one of x, (x,), (x, y)
    """
    if isinstance(data, list):
        data = tuple(data)
    if not isinstance(data, tuple):
        return data, None
    elif 0 < len(data) <= 2:
        representation = [None, None]
        representation[:len(data)] = data
        return representation
    else:
        msg = (
            "Data is expected to be in format x, (x,), (x, y), received: {}"
        ).format(data)
        raise ValueError(msg)
