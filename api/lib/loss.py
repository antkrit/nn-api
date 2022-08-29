"""Contains simple loss functions."""
from api.lib.autograd import math


__all__ = ('mse', 'crossentropy')


def mse(predictions, targets):
    """Mean squared error"""
    pow_ = (predictions - targets) ** 2
    return math.mean(pow_)


def crossentropy(predictions, targets):
    """Categorical crossentropy"""
    mul_ = targets * math.log(predictions)
    sum_ = math.sum(mul_, axis=1)
    return -math.sum(sum_)
