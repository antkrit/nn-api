"""Contains objects to work with data."""
from collections.abc import MutableMapping

import numpy as np


class Dataset:
    """Collection of data.

    Implemented as an infinite generator. Use `__len__` property to find out
    number of unique batches in one dataset.

    :param x_data: input data
    :param y_data: input data
    :param batch_size: number of samples per one batch, defaults to 1
    :param x_dim: x_data shape, defaults to (1, 1)
    :param y_dim: y_data shape, if None - will be equal to np.ones_like(dim),
        defaults to None
    :param shuffle: whether to shuffle data on epoch end
    """

    def __init__(
        self,
        x_data,
        y_data,
        batch_size=1,
        x_dim=(1, 1),
        y_dim=None,
        shuffle=True,
    ):
        """Constructor method."""
        self.x = x_data
        self.y = y_data

        if len(self.x) != len(self.y):
            raise ValueError(
                f"Cannot match the lengths: {len(self.x)} and {len(self.y)}."
            )

        self.shuffle = shuffle
        self.indexes = np.arange(len(self.x))
        if self.shuffle:
            np.random.shuffle(self.indexes)

        self.batch_size = min(batch_size, self.indexes.size)
        self.x_dim = np.atleast_1d(x_dim)
        self.y_dim = None

        if y_dim is not None:
            self.y_dim = np.atleast_1d(y_dim)

        self.__max = self.__len__()
        self.__inner_state = 0

    def __len__(self):
        return max(len(self.x) // max(1, self.batch_size), 1)

    def __getitem__(self, index):
        """Generate batch of data."""
        if self.__inner_state >= self.__max:
            raise IndexError(
                f"Index {index} out of range for {self.__max} batches."
            )

        indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        x, y = self.__generate_data(indexes)

        return x, y

    def __iter__(self):
        self.__inner_state = 0
        return self

    def __next__(self):
        if self.__inner_state >= self.__max:
            self.on_epoch_end()
            self.__inner_state = 0

        result = self.__getitem__(self.__inner_state)

        self.__inner_state += 1
        return result

    def on_epoch_end(self):
        """Do some postprocessing at the end of each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __generate_data(self, ids):
        """Generate data containing batch_size samples."""
        if self.y_dim is None:
            y_dim = np.ones_like(self.x_dim)
        else:
            y_dim = self.y_dim

        x = np.empty((self.batch_size, *self.x_dim))
        y = np.empty((self.batch_size, *y_dim))

        # TODO: instead of getting data by index, load it with some method
        # this works only for arrays of data. Need to make this procedure lazy
        for i, id_ in enumerate(ids):
            x[i,] = self.x[id_]
            y[i,] = self.y[id_]

        return x, y


class Container(MutableMapping):
    """Base dict-like container class.

    To get an object use any of the three options

    .. code::

        >>> container = Container(name=..., obj=3)
        >>> container['obj']
        3
        >>> container.obj
        3
        >>> container('obj')
        3

    To get the compiled instance - use __call__ method

    .. code-block:: python

        >>> container = Container(name=..., obj=lambda x: x)
        >>> container('obj', compiled=True, x=3)
        3

    .. note::
        if in `__call__()` the first argument is not str, then this
        object will be returned.
    """

    def __init__(self, name, *args, **kwargs):
        """Constructor method."""
        self.name = name
        self.store = {}
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        return self.store[key]

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __call__(self, obj_name, *args, compiled=False, **kwargs):
        if not isinstance(obj_name, str):
            return obj_name

        obj = self.__getitem__(key=obj_name)
        if callable(obj) and compiled:
            return obj(*args, **kwargs)

        return obj

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __repr__(self):
        return f"Container-{self.name}({self.store.items()})"


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
    if 0 < len(data) <= 2:
        representation = [None, None]
        representation[: len(data)] = data
        return representation

    msg = (
        "Data is expected to be in format x, (x,), (x, y)," f" received: {data}"
    )
    raise ValueError(msg)
