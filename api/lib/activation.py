"""Contains simple activation functions."""
from api.lib.autograd import math


__all__ = ('tanh', 'relu', 'sigmoid', 'gaussian', 'softmax')


def tanh(x):
    """Hyperbolic tangent."""
    num = (math.exp(x) - math.exp(-x))
    denom = (math.exp(x) + math.exp(-x))
    return num / denom


def relu(x):
    """Rectified linear activation."""
    return math.max(x, 0)


def sigmoid(x):
    """Logistic(sigmoid) activation."""
    return 1 / (1 + math.exp(-x))


def gaussian(x):
    """Gaussian function"""
    return math.exp(-(x*x))


def softmax(x):
    """Normalized exponential function"""
    num = math.exp(x)
    denom = math.sum(math.exp(x), axis=1)
    return num / denom
