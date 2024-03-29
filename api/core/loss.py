"""Contains implementation of commonly used loss functions.

Regression:

- (Root) Mean Squared Error
- Mean Absolute Error
- Mean Bias Error
- Huber Loss

Binary Classification:

- Likelihood Loss
- Binary Cross Entropy
- (Squared) Hinge Loss

Multinomial Classification:

- Categorical Cross Entropy
- Kullback-Leibler Divergence
"""
import abc

import numpy as np

from api.core import autograd as ag

__all__ = (
    "BaseLoss",
    "MSE",
    "MAE",
    "MBE",
    "Huber",
    "LHL",
    "BCE",
    "Hinge",
    "CCE",
    "KLD",
)


class BaseLoss:
    """Base loss class.

    :param name: name of the loss, defaults to loss
    :param session: current session, if None - creates new, defaults to None
    :param threshold: some minute value to avoid problems like
        div by 0 or log(0), defaults to 0
    """

    def __init__(self, name="loss", session=None, threshold=0):
        self.session = session or ag.Session()
        self.threshold = threshold
        self.name = name

        self._value = None

    @property
    def value(self):
        """Return loss value."""
        if hasattr(self._value, "value"):
            # if loss value is Node return its value
            return self._value.value
        return self._value

    @abc.abstractmethod
    def forward(self, y_pred, y_true):
        """Calculate loss."""
        raise NotImplementedError("Must be implemented in child classes.")

    def __call__(self, y_pred, y_true, *args, **kwargs):
        self._value = ag.utils.node_wrapper(
            self.forward, y_pred, y_true, *args, **kwargs
        )
        return self._value

    def __str__(self):
        return f"{self.name}: {self.value}"


class MSE(BaseLoss):
    """(Root) Mean Squared Error.

    Finds the averaged squared difference between predicted and
    actual values. Mostly used for regression problems.

    :param root: set to True when RMSE loss is required, defaults to False
    """

    def __init__(
        self, name="mean_squared_error", session=None, root=False, threshold=0
    ):
        """Constructor method."""
        super().__init__(name=name, session=session, threshold=threshold)

        self.root = root
        if root:
            self.name = "root_" + self.name

    def forward(self, y_pred, y_true):
        """Calculate (R)MSE."""
        err = (y_pred - y_true) ** 2
        total = ag.ops.mean(err)
        return ag.ops.sqrt(total) if self.root else total


class MAE(BaseLoss):
    """Mean Absolute Error.

    Finds the averaged absolute difference between predicted and
    actual values. Mostly used for regression problems.
    """

    def __init__(self, name="mean_absolute_error", session=None, threshold=0):
        """Constructor method."""
        super().__init__(name=name, session=session, threshold=threshold)

    def forward(self, y_pred, y_true):
        """Calculate MAE."""
        err = ag.ops.abs(y_pred - y_true)
        return ag.ops.mean(err)


class MBE(BaseLoss):
    """Mean Bias Error.

    Finds the averaged difference between predicted and actual values.
    Mostly used for regression problems.
    """

    def __init__(self, name="mean_bias_error", session=None, threshold=0):
        """Constructor method."""
        super().__init__(name=name, session=session, threshold=threshold)

    def forward(self, y_pred, y_true):
        """Calculate MBE."""
        err = y_pred - y_true
        return ag.ops.mean(err)


class Huber(BaseLoss):
    """Huber Loss.

    Combination of :class:`MAE` and :class:`MSE`. This loss applies MAE for
    values that fits under the expected values, and MSE is applied to outliers.

    :param delta: a number, point where two functions are connected,
        defaults to 1
    """

    def __init__(self, name="huber", delta=1, session=None, threshold=0):
        """Constructor method."""
        super().__init__(name=name, session=session, threshold=threshold)
        self.delta = delta

    def forward(self, y_pred, y_true):
        """Calculate Huber loss."""
        err = ag.ops.abs(y_pred - y_true)

        is_err_small = self.session.run(err) <= self.delta
        # implementation details: initially, the Huber loss is a system with
        # 2 equations, so it's possible to calculate all 'if' conditions in
        # advance, multiply it and its inverted version by the system
        # inequalities and then just add them
        cond_true = ag.utils.convert_to_node(
            value=np.asarray(is_err_small, dtype=int)
        )
        cond_false = ag.utils.convert_to_node(
            value=np.asarray(np.invert(is_err_small), dtype=int)
        )

        lss_cond_true = cond_true * 0.5 * err**2
        lss_cond_false = cond_false * self.delta * (err - 0.5 * self.delta)
        return ag.ops.mean(lss_cond_true + lss_cond_false)


class LHL(BaseLoss):
    """Likelihood Loss.

    Multiplication of the probability for the ground truth labels. If the
    true class is [1], we use the output probability, otherwise if the
    true class is [0], we use 1-output probability. Mostly used for
    binary classification.
    """

    def __init__(self, name="likelihood", session=None, threshold=0):
        """Constructor method."""
        super().__init__(name=name, session=session, threshold=threshold)

    def forward(self, y_pred, y_true):
        """Calculate LHL."""
        err = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return ag.ops.mean(err)


class BCE(BaseLoss):
    """Binary Cross Entropy.

    Modification of the :class:`LHL` using logarithms. This allows for
    more severe penalties for large errors.
    """

    def __init__(
        self, name="binary_cross_entropy", session=None, threshold=1e-32
    ):
        """Constructor method."""
        super().__init__(name=name, session=session, threshold=threshold)

    def forward(self, y_pred, y_true):
        """Calculate BCE."""
        err = y_true * ag.ops.log10(y_pred + self.threshold)
        err_1 = (1 - y_true) * ag.ops.log10(1 - y_pred + self.threshold)
        return ag.ops.mean(-(err + err_1))


class Hinge(BaseLoss):
    """Hinge Loss.

    Primarily was developed to evaluate SVM models. Incorrect or less
    confident right predictions are more penalized with Hinge Loss

    .. warning::
        To apply Hinge Loss, classes must be marked as 1 and -1 (not 0)
    """

    def __init__(self, name="hinge", session=None, threshold=0):
        """Constructor method."""
        super().__init__(name=name, session=session, threshold=threshold)

    def forward(self, y_pred, y_true):
        """Calculate HL."""
        err = ag.ops.max(0, 1 - (y_pred * y_true))
        return ag.ops.mean(err)


class CCE(BaseLoss):
    """Categorical Cross Entropy.

    Used for multinomial classification. Similar formula as in the :class:`BCE`,
    but with one extra step. The output label is assigned one-hot category
    encoding value in form of 0s and 1. Calculate the loss for every pair
    of targets and predictions and return the mean
    """

    def __init__(
        self, name="categorical_cross_entropy", session=None, threshold=1e-32
    ):
        """Constructor method."""
        super().__init__(name=name, session=session, threshold=threshold)

    def forward(self, y_pred, y_true):
        """Calculate CCE."""
        err = y_true * ag.ops.log10(y_pred + self.threshold)
        return ag.ops.mean(-err)


class KLD(BaseLoss):
    """Kullback-Leibler Divergence.

    Similar to :class:`CCE`, but considers the probability of occurrence
    of observations. Useful when classes are not balanced.
    """

    def __init__(
        self, name="kullback_leibler_divergence", session=None, threshold=1e-32
    ):
        """Constructor method."""
        super().__init__(name=name, session=session, threshold=threshold)

    def forward(self, y_pred, y_true):
        """Calculate KLD."""
        prob = (y_true + self.threshold) / (y_pred + self.threshold)
        err = y_true * ag.ops.log10(prob)
        return ag.ops.mean(err)
