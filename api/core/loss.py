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


__all__ = ('MSE', 'MAE', 'MBE', 'Huber', 'LHL', 'BCE', 'Hinge', 'CCE', 'KLD')


class BaseLoss:
    """Base loss class.

    :param threshold: some minute value to avoid problems like
        div by 0 or log(0), defaults to 0
    :param session: current session, if None - creates new, defaults to None
    """

    def __init__(self, session=None, threshold=0):
        self.session = session or ag.Session()
        self.threshold = threshold

    @abc.abstractmethod
    def forward(self, y_true, y_pred):
        """Calculate loss."""
        raise NotImplementedError("Must be implemented in child classes.")

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in child classes.")


class MSE(BaseLoss):
    """(Root) Mean Squared Error.

    :param root: set to True when RMSE loss is required, defaults to False
    """

    def __init__(self, session=None, root=False, threshold=0):
        """Constructor method."""
        super().__init__(session, threshold)
        self.root = root

    def forward(self, y_pred, y_true):
        """Calculate (R)MSE.

        Finds the averaged squared difference between predicted and
        actual values. Mostly used for regression problems.

        :param y_pred: values predicted by model
        :param y_true: actual values
        :return: node that contains (root) mean squared error
        """
        err = (y_pred - y_true) ** 2
        total = ag.ops.mean(err)
        return ag.ops.sqrt(total) if self.root else total

    def __call__(self, y_pred, y_true, *args, **kwargs):
        return ag.utils.node_wrapper(
            self.forward,
            y_pred, y_true,
            *args, **kwargs
        )


class MAE(BaseLoss):
    """Mean Absolute Error."""

    def __init__(self, session=None, threshold=0):
        """Constructor method."""
        super().__init__(session, threshold)

    def forward(self, y_pred, y_true):
        """Calculate MAE.

        Finds the averaged absolute difference between predicted and
        actual values. Mostly used for regression problems.

        :param y_pred: values predicted by model
        :param y_true: actual values
        :return: node that contains mean absolute error
        """
        err = ag.ops.abs(y_pred - y_true)
        return ag.ops.mean(err)

    def __call__(self, y_pred, y_true, *args, **kwargs):
        return ag.utils.node_wrapper(
            self.forward,
            y_pred, y_true,
            *args, **kwargs
        )


class MBE(BaseLoss):
    """Mean Bias Error."""

    def __init__(self, session=None, threshold=0):
        """Constructor method."""
        super().__init__(session, threshold)

    def forward(self, y_pred, y_true):
        """Calculate MBE.

        Finds the averaged difference between predicted and actual values.
        Mostly used for regression problems.

        :param y_pred: values predicted by model
        :param y_true: actual values
        :return: node that contains mean absolute error
        """
        err = y_pred - y_true
        return ag.ops.mean(err)

    def __call__(self, y_pred, y_true, *args, **kwargs):
        return ag.utils.node_wrapper(
            self.forward,
            y_pred, y_true,
            *args, **kwargs
        )


class Huber(BaseLoss):
    """Huber Loss.

    Combination of class:`MAE` and class:`MSE`. This loss applies MAE for values
    that fits under the expected values, and MSE is applied to outliers.

    :param delta: a number, point where two functions are connected,
        defaults to 1
    """

    def __init__(self, delta=1, session=None, threshold=0):
        """Constructor method."""
        super().__init__(session, threshold)
        self.delta = delta

    def forward(self, y_pred, y_true):
        """Calculate Huber loss.

        :param y_pred: values predicted by model
        :param y_true: actual values
        :return: node that contains mean absolute error
        """
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

        lss_cond_true = cond_true * 0.5 * err ** 2
        lss_cond_false = cond_false * self.delta * (err - 0.5 * self.delta)
        return ag.ops.mean(lss_cond_true+lss_cond_false)

    def __call__(self, y_pred, y_true, *args, **kwargs):
        return ag.utils.node_wrapper(
            self.forward,
            y_pred, y_true,
            *args, **kwargs
        )


class LHL(BaseLoss):
    """Likelihood Loss.

    Multiplication of the probability for the ground truth labels. If the
    true class is [1], we use the output probability, otherwise if the
    true class is [0], we use 1-output probability. Mostly used for
    binary classification.
    """

    def __init__(self, session=None, threshold=0):
        """Constructor method."""
        super().__init__(session, threshold)

    def forward(self, y_pred, y_true):
        """Calculate LHL.

        :param y_pred: values predicted by model
        :param y_true: actual values
        :return: node that contains likelihood loss
        """
        err = (y_true * y_pred + (1 - y_true) * (1 - y_pred))
        return ag.ops.mean(err)

    def __call__(self, y_pred, y_true, *args, **kwargs):
        return ag.utils.node_wrapper(
            self.forward,
            y_pred, y_true,
            *args, **kwargs
        )


class BCE(BaseLoss):
    """Binary Cross Entropy.

    Modification of the class:`LHL` using logarithms. This allows for
    more severe penalties for large errors.
    """

    def __init__(self, session=None, threshold=1e-32):
        """Constructor method."""
        super().__init__(session, threshold)

    def forward(self, y_pred, y_true):
        """Calculate BCE.

        :param y_pred: values predicted by model
        :param y_true: actual values
        :return: node that contains binary cross entropy loss
        """
        err = y_true * ag.ops.log10(y_pred + self.threshold)
        err_1 = (1 - y_true) * ag.ops.log10(1 - y_pred + self.threshold)
        return ag.ops.mean(-(err+err_1))

    def __call__(self, y_pred, y_true, *args, **kwargs):
        return ag.utils.node_wrapper(
            self.forward,
            y_pred, y_true,
            *args, **kwargs
        )


class Hinge(BaseLoss):
    """Hinge Loss.

    Primarily was developed to evaluate SVM models. Incorrect or less
    confident right predictions are more penalized with Hinge Loss

    .. warning::
        To apply Hinge Loss, classes must be marked as 1 and -1 (not 0)
    """

    def __init__(self, session=None, threshold=0):
        """Constructor method."""
        super().__init__(session, threshold)

    def forward(self, y_pred, y_true):
        """Calculate HL.

        :param y_pred: values predicted by model
        :param y_true: actual values
        :return: node, that contains (squared) hinge loss
        """
        err = ag.ops.max(0, 1 - (y_pred * y_true))
        return ag.ops.mean(err)

    def __call__(self, y_pred, y_true, *args, **kwargs):
        return ag.utils.node_wrapper(
            self.forward,
            y_pred, y_true,
            *args, **kwargs
        )


class CCE(BaseLoss):
    """Categorical Cross Entropy.

    Used for multinomial classification. Similar formula as in the class:`BSE`,
    but with one extra step. The output label is assigned one-hot category
    encoding value in form of 0s and 1. Calculate the loss for every pair
    of targets and predictions and return the mean
    """

    def __init__(self, session=None, threshold=1e-32):
        """Constructor method."""
        super().__init__(session, threshold)

    def forward(self, y_pred, y_true):
        """Calculate CCE.

        :param y_pred: values predicted by model
        :param y_true: actual values
        :return: node, that contains categorical cross entropy loss
        """
        err = y_true * ag.ops.log10(y_pred + self.threshold)
        return ag.ops.mean(-err)

    def __call__(self, y_pred, y_true, *args, **kwargs):
        return ag.utils.node_wrapper(
            self.forward,
            y_pred, y_true,
            *args, **kwargs
        )


class KLD(BaseLoss):
    """Kullback-Leibler Divergence.

    Similar to class:`CCE`, but considers the probability of occurrence
    of observations. Useful when classes are not balanced.
    """

    def __init__(self, session=None, threshold=1e-32):
        """Constructor method."""
        super().__init__(session, threshold)

    def forward(self, y_pred, y_true):
        """Calculate KLD.

        :param y_pred: values predicted by model
        :param y_true: actual values
        :return: node, that contains Kullback-Leibler loss
        """
        prob = (y_true + self.threshold) / (y_pred + self.threshold)
        err = y_true * ag.ops.log10(prob)
        return ag.ops.mean(err)

    def __call__(self, y_pred, y_true, *args, **kwargs):
        return ag.utils.node_wrapper(
            self.forward,
            y_pred, y_true,
            *args, **kwargs
        )
