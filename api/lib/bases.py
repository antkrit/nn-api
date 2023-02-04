"""Contains various base classes."""
import api.lib.autograd as ag
from operator import itemgetter


class BaseLayer:
    """Base layer class."""

    def forward(self, x, *args, **kwargs):
        """Calculate output of the layer."""
        raise NotImplementedError("Must be implemented in child classes.")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in child classes.")


class BaseActivation:
    """Base activation class.

    :param threshold: some minute value to avoid problems like
        div by 0 or log(0), defaults to 0
    :param session: current session, if None - creates new, defaults to None
    """

    def __init__(self, session=None, threshold=0):
        self.session = session or ag.Session()
        self.threshold = threshold

    def forward(self, x):
        """Calculate function."""
        raise NotImplementedError("Must be implemented in child classes.")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in child classes.")


class BaseLoss:
    """Base loss class.

    :param threshold: some minute value to avoid problems like
        div by 0 or log(0), defaults to 0
    :param session: current session, if None - creates new, defaults to None
    """

    def __init__(self, session=None, threshold=0):
        self.session = session or ag.Session()
        self.threshold = threshold

    def forward(self, y_true, y_pred):
        """Calculate loss."""
        raise NotImplementedError("Must be implemented in child classes.")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in child classes.")


class BaseInitializer:
    """Base Initializer class.

    :param seed: number in the range [0, 2**32], define the internal state of
        the generator so that random results can be reproduced, defaults to None
    """

    def __init__(self, seed):
        self.seed = seed

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in a subclass.")


class BaseOptimizer:
    """Base optimizer class.

    :param session: current session, if None - creates new, defaults to None
    :param trainable_variables: variables to optimize, defaults to
        weight and bias
    """

    def __init__(self, trainable_variables, session=None):
        self.session = session or ag.Session()
        self.trainable = trainable_variables
        self._itemgetter = itemgetter(*self.trainable)

    def apply_gradient(self, *args, **kwargs):
        """Apply computed gradients to trainable variables."""
        raise NotImplementedError("Must be implemented in subclasses.")

    def minimize(self, *args, **kwargs):
        """Combine gradient computing and applying."""
        raise NotImplementedError("Must be implemented in subclasses.")
