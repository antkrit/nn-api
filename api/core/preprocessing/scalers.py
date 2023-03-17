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

    @abc.abstractmethod
    def fit_transform(self, *args, **kwargs):
        """Combine fit and transform methods."""
        raise NotImplementedError("Must be implemented in subclasses.")


class MinMaxScaler(BaseScaler):

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

    def fit_transform(self, sample):
        """Combine fit and transform methods."""
        return self.fit(sample).transform(sample)


class StandardScaler(BaseScaler):

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

    def fit_transform(self, sample):
        """Combine fit and transform methods."""
        return self.fit(sample).transform(sample)
