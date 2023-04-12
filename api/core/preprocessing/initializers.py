"""Contains simple functions for generating arrays with numbers."""
import abc

import numpy as np

from api.core.autograd.utils import convert_to_node

__all__ = (
    "NormalInitializer",
    "UniformInitializer",
    "zeros",
    "ones",
    "he_normal",
    "he_uniform",
    "random_normal",
    "random_uniform",
    "xavier_normal",
    "xavier_uniform",
    "lecun_normal",
    "lecun_uniform",
)


class BaseInitializer:
    """Base Initializer class.

    :param seed: number in the range [0, 2**32], define the internal state of
        the generator so that random results can be reproduced, defaults to None
    """

    def __init__(self, seed):
        self.seed = seed

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in subclasses.")


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

        self.__worker = np.random.randn

    def __call__(self, size, *args, **kwargs):
        if self.seed is not None:
            np.random.seed(self.seed)

        ndist = self.__worker(*size)

        return convert_to_node(
            value=self.sigma * ndist + self.mu, *args, **kwargs
        )


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

        self.__worker = np.random.uniform

    def __call__(self, size, *args, **kwargs):
        if self.seed is not None:
            np.random.seed(self.seed)

        ndist = self.__worker(*self.bounds, size=size)

        return convert_to_node(value=ndist, *args, **kwargs)


def zeros(size, *args, **kwargs):
    """Get array of zeros."""
    seed = kwargs.pop("seed", None)
    return NormalInitializer(seed=seed)(size, *args, **kwargs)


def ones(size, *args, **kwargs):
    """Get array of ones."""
    return NormalInitializer(mu=1, sigma=0, seed=kwargs.pop("seed", None))(
        size, *args, **kwargs
    )


def random_normal(size, *args, **kwargs):
    """Get random array from the standard normal distribution."""
    return NormalInitializer(mu=0, sigma=1, seed=kwargs.pop("seed", None))(
        size, *args, **kwargs
    )


def random_uniform(size, *args, **kwargs):
    """Get random array from the uniform distribution in the range [-1, 1]."""
    return UniformInitializer(low=-1, high=1, seed=kwargs.pop("seed", None))(
        size, *args, **kwargs
    )


def xavier_normal(size, *args, **kwargs):
    """Xavier (or Glorot) normal initialization."""
    n_in, n_out = size[-2:]
    distribution = np.sqrt(2 / (n_in + n_out))
    return NormalInitializer(
        mu=0, sigma=distribution, seed=kwargs.pop("seed", None)
    )(size, *args, **kwargs)


def xavier_uniform(size, *args, **kwargs):
    """Xavier (or Glorot) uniform initialization."""
    n_in, n_out = size[-2:]
    limit = np.sqrt(6 / (n_in + n_out))
    return UniformInitializer(
        low=-limit, high=limit, seed=kwargs.pop("seed", None)
    )(size, *args, **kwargs)


def he_normal(size, *args, **kwargs):
    """Kaiming(or He) normal initialization."""
    n_in = size[-2]
    distribution = np.sqrt(2 / n_in)
    return NormalInitializer(
        mu=0, sigma=distribution, seed=kwargs.pop("seed", None)
    )(size, *args, **kwargs)


def he_uniform(size, *args, **kwargs):
    """Kaiming(or He) uniform initialization."""
    n_in = size[-2]
    limit = np.sqrt(6 / n_in)
    return UniformInitializer(
        low=-limit, high=limit, seed=kwargs.pop("seed", None)
    )(size, *args, **kwargs)


def lecun_normal(size, *args, **kwargs):
    """LeCun normal initialization."""
    n_in = size[-2]
    distribution = np.sqrt(1 / n_in)
    return NormalInitializer(
        mu=0, sigma=distribution, seed=kwargs.pop("seed", None)
    )(size, *args, **kwargs)


def lecun_uniform(size, *args, **kwargs):
    """LeCun uniform initialization."""
    n_in = size[-2]
    limit = np.sqrt(3 / n_in)

    return UniformInitializer(
        low=-limit, high=limit, seed=kwargs.pop("seed", None)
    )(size, *args, **kwargs)
