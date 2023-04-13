"""Contains implementation for data scalers."""
import abc

import numpy as np


class BaseScaler:
    """Base scaler class."""

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """Get scaler parameters from data sample."""
        raise NotImplementedError("Must be implemented in subclasses.")

    @abc.abstractmethod
    def transform(self, *args, **kwargs):
        """Transform data using scaler parameters."""
        raise NotImplementedError("Must be implemented in subclasses.")

    def fit_transform(self, sample):
        """Combine fit and transform methods."""
        return self.fit(sample).transform(sample)


class MinMaxScaler(BaseScaler):
    """Transform features by scaling each feature to a given range.

    The scaled x is given by (high and low values are taken from the
    feature range):

    .. code-block:: python

        std = (x - min) / (max - min)
        x_ = std * (high - low) + low

    The min and max parameters can be specified manually or using the
    ``MinMaxScaler.fit()`` method. It is also possible to fit and transform
    at the same time (see ``BaseScaler.fit_transform()`` implementation). After
    the `fit` method has been explicitly or implicitly called, the sample info
    (min and max) are saved and can be applied to another sample (using bare
    ``transform()`` method).

    .. code-block::

        >>> data = [[0, 1], [1, 0], [1, 1], [0, 0]]
        >>> mms = MinMaxScaler()
        >>> mms.fit(data)
        MinMaxScaler()
        >>> print(mms.min_)
        [0 0]
        >>> print(mms.max_)
        [1 1]
        >>> mms.transform(data)
        array([[0., 1.],
               [1., 0.],
               [1., 1.],
               [0., 0.]])

    :param feature_range: the range to which the data is to be scaled,
        defaults to (0, 1)
    :param sample_min: sample minimum (axis=0)
    :param sample_max: sample maximum (axis=0)
    """

    def __init__(self, feature_range=(0, 1), sample_min=None, sample_max=None):
        """Constructor method."""
        self.low, self.high = feature_range
        self.min_ = sample_min
        self.max_ = sample_max

    def fit(self, sample):
        """Get scaler parameters from data sample."""
        self.min_ = np.amin(sample, axis=0)
        self.max_ = np.amax(sample, axis=0)
        return self

    def transform(self, sample):
        """Transform data using scaler parameters."""
        sample_std = (sample - self.min_) / (self.max_ - self.min_)
        return sample_std * (self.high - self.low) + self.low

    def __repr__(self):
        return "MinMaxScaler()"


class StandardScaler(BaseScaler):
    """Standardize data by removing the mean and scaling to unit variance.

    The standard x* is calculated as (x - mu) / sigma

    The mu and sigma parameters can be specified manually or using the
    `StandardScaler.fit()` method. It is also possible to fit and transform
    at the same time (see `BaseScaler.fit_transform()` implementation). After
    the `fit` method has been explicitly or implicitly called, the sample info
    (mu and sigma) are saved and can be applied to another sample (using
    bare `transform()` method).

    .. code-block::

        >>> data = [[0, 1], [1, 0], [1, 1], [0, 0]]
        >>> sc = StandardScaler()
        >>> sc.fit(data)
        StandardScaler()
        >>> print(sc.mu)
        [0.5 0.5]
        >>> print(sc.sigma)
        [0.5 0.5]
        >>> sc.transform(data)
        array([[-1.,  1.],
               [ 1., -1.],
               [ 1.,  1.],
               [-1., -1.]])

    :param mu: mean of the sample
    :param sigma: standard deviation of the sample
    """

    def __init__(self, mu=None, sigma=None):
        """Constructor method."""
        self.mu = mu
        self.sigma = sigma

    def fit(self, sample):
        """Get scaler parameters from data sample."""
        self.mu = np.mean(sample, axis=0)
        self.sigma = np.std(sample - self.mu, axis=0)
        return self

    def transform(self, sample):
        """Transform data using scaler parameters."""
        return (sample - self.mu) / self.sigma

    def __repr__(self):
        return "StandardScaler()"
