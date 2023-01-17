"""Contains simple activation functions."""
import api.lib.autograd as ag


__all__ = ('tanh', 'relu', 'sigmoid', 'gaussian', 'softmax')


def tanh(x):
    """Hyperbolic tangent."""
    num = (ag.exp(x) - ag.exp(-x))
    denom = (ag.exp(x) + ag.exp(-x))
    return num / denom


def relu(x):
    """Rectified linear activation."""
    return ag.max(x, 0)


def sigmoid(x):
    """Logistic(sigmoid) activation."""
    return 1 / (1 + ag.exp(-x))


def gaussian(x):
    """Gaussian function"""
    return ag.exp(-(x*x))


def softmax(x):
    """Normalized exponential function"""
    num = ag.exp(x)
    denom = ag.sum(ag.exp(x), axis=1)
    return num / denom
