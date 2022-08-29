"""Contains simple functions for generating arrays with numbers."""
import numpy as np
from api.lib.autograd import Variable


__all__ = (
    'zeros_initializer',
    'ones_initializer',
    'rnormal_initializer',
    'xavier_initializer',
    'kaiming_initializer'
)


def zeros_initializer(shape, *args, **kwargs):
    """Get array of zeros.

    :param shape: size of the array
    """
    return Variable(np.zeros(shape, *args, **kwargs))


def ones_initializer(shape, *args, **kwargs):
    """Get array of ones.

    :param shape: size of the array
    """
    return Variable(np.ones(shape, *args, **kwargs))


def rnormal_initializer(shape):
    """Get random array from the "standard normal" distribution.

    :param shape: size of the array
    """
    return Variable(np.random.randn(*shape))


def xavier_initializer(shape):
    """Xavier(or Glorot) initialization.

    :param shape: size of the array
    """
    return Variable(
        np.random.randn(*shape) * np.sqrt(6/np.sum(shape))
    )


def kaiming_initializer(shape):
    """Kaiming(or He) initialization.

    :param shape: size of the array
    """
    return Variable(
        np.random.randn(*shape) * np.sqrt(2 / shape[0])
    )
