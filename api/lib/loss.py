"""Contains simple loss functions."""
import api.lib.autograd as ag


__all__ = ('mse', 'crossentropy')


def mse(predictions, targets):
    """Mean squared error"""
    pow_ = (predictions - targets) ** 2
    return ag.mean(pow_)


def crossentropy(predictions, targets):
    """Categorical crossentropy"""
    mul_ = targets * ag.log(predictions)
    sum_ = ag.sum(mul_, axis=1)
    return -ag.sum(sum_)
