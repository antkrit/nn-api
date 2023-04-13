"""Contains utility functions used for models."""
from functools import wraps

from api.core import namespace


def control_compile(func):
    """Check for decorated methods if the model is compiled.

    :raises ModelIsNotCompiledException: if method was run,
        but model is not compiled
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.built:
            raise namespace.exceptions.ModelIsNotCompiled(
                "Model must be compiled first."
            )
        return func(self, *args, **kwargs)

    return wrapper
