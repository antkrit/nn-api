"""Contains useful frequently used objects."""
import numpy as np


def atleast_3d(array):
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
    if len(array.shape) == 1:
        return array[np.newaxis, :, np.newaxis]
    if len(array.shape) == 2:
        return array[np.newaxis, :, :]

    return array
