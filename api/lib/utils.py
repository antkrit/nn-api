"""Contains useful frequently used objects."""
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