"""Contains simple functions for generating arrays with numbers."""
import numpy as np
from api.lib.bases import BaseInitializer
from api.lib.autograd import Variable


__all__ = (
    'NormalInitializer',
    'UniformInitializer',
    'zeros',
    'ones',
    'he_normal',
    'he_uniform',
    'random_normal',
    'random_uniform',
    'xavier_normal',
    'xavier_uniform',
    'lecun_normal',
    'lecun_uniform'
)


class NormalInitializer(BaseInitializer):
    """Normal distribution initializer class N(m,σ).

    Generates random samples from the normal distribution with mean `mu`
    and standard deviation `sigma`, using sigma*randn(in, out) + mu formula.
    Specific implementations of this initializer can change these two
    parameters. Default configuration will generate simple zeros vector.

    :param sigma: standard deviation of the normal distribution, defaults to 0
    :param mu: float, mean of the normal distribution, defaults to 0
    """
    def __init__(self, mu=0, sigma=0, seed=None):
        """Constructor method."""
        super().__init__(seed)

        self.mu = mu
        self.sigma = sigma

        self.__base_initializer = np.random.randn

    def __call__(self, size, *args, **kwargs):
        np.random.seed(self.seed)
        ndist = self.__base_initializer(*size)
        return Variable(self.sigma*ndist + self.mu, *args, **kwargs)


class UniformInitializer(BaseInitializer):
    """Uniform distribution initializer class U(low, high).

    Generates random samples from the uniform distribution in the range
    [low, high]. Specific implementations of this initializer can change these
    two parameters. Default configuration will generate simple zeros vector.

    :param low: lower boundary of the output interval, defaults to 0
    :param high: upper boundary of the output interval, defaults to 0
    """
    def __init__(self, low=0, high=0, seed=None):
        """Constructor method."""
        super().__init__(seed)

        self.low = low
        self.high = high
        self.bounds = (low, high)

        self.__base_initializer = np.random.uniform

    def __call__(self, size, *args, **kwargs):
        np.random.seed(self.seed)
        ndist = self.__base_initializer(*self.bounds, size=size)
        return Variable(ndist, *args, **kwargs)


def zeros(n_in, n_out, *args, **kwargs):
    """Get array of zeros."""
    return NormalInitializer(*args, **kwargs)((n_in, n_out))


def ones(n_in, n_out, *args, **kwargs):
    """Get array of ones."""
    return NormalInitializer(mu=1, sigma=0, *args, **kwargs)((n_in, n_out))


def random_normal(n_in, n_out, *args, **kwargs):
    """Get random array from the standard normal distribution."""
    return NormalInitializer(mu=0, sigma=1, *args, **kwargs)((n_in, n_out))


def random_uniform(n_in, n_out, *args, **kwargs):
    """Get random array from the uniform distribution in the range [-1, 1]."""
    return UniformInitializer(low=-1, high=1, *args, **kwargs)((n_in, n_out))


def xavier_normal(n_in, n_out, *args, **kwargs):
    """Xavier (or Glorot) normal initialization."""
    distribution = np.sqrt(2/(n_in+n_out))
    return NormalInitializer(
        mu=0, sigma=distribution,
        *args, **kwargs
    )((n_in, n_out))


def xavier_uniform(n_in, n_out, *args, **kwargs):
    """Xavier (or Glorot) uniform initialization."""
    limit = np.sqrt(6/(n_in+n_out))
    return UniformInitializer(
        low=-limit, high=limit,
        *args, **kwargs
    )((n_in, n_out))


def he_normal(n_in, n_out, *args, **kwargs):
    """Kaiming(or He) normal initialization."""
    distribution = np.sqrt(2 / n_in)
    return NormalInitializer(
        mu=0, sigma=distribution,
        *args, **kwargs
    )((n_in, n_out))


def he_uniform(n_in, n_out, *args, **kwargs):
    """Kaiming(or He) uniform initialization."""
    limit = np.sqrt(6 / n_in)
    return UniformInitializer(
        low=-limit, high=limit,
        *args, **kwargs
    )((n_in, n_out))


def lecun_normal(n_in, n_out, *args, **kwargs):
    """LeCun normal initialization."""
    distribution = np.sqrt(1 / n_in)
    return NormalInitializer(
        mu=0, sigma=distribution,
        *args, **kwargs
    )((n_in, n_out))


def lecun_uniform(n_in, n_out, *args, **kwargs):
    """LeCun uniform initialization."""
    limit = np.sqrt(3 / n_in)
    return UniformInitializer(
        low=-limit, high=limit,
        *args, **kwargs
    )((n_in, n_out))
