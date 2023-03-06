"""Contains various base classes."""
from api.lib.exception import NoGradientException
from api.lib.autograd import Session, Operation, utils


class BaseLayer:
    """Base layer class."""

    def __init__(self, session=None):
        """Constructor method."""
        self.trainable = []
        self.session = session or Session()

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
        self.session = session or Session()
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
        self.session = session or Session()
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


class BaseOptimizer(Operation):
    """Base optimizer class.

    :param session: current session, if None - creates new, defaults to None
    :param trainable_variables: variables to optimize, defaults to
        weight and bias
    """

    def __init__(self, trainable_variables, session=None, name=None):
        """Constructor method."""
        super().__init__(name=name)
        self.session = session or Session()
        self.trainable = trainable_variables

    def apply_gradient(self, *args, **kwargs):
        """Apply computed gradients to trainable variables."""
        raise NotImplementedError("Must be implemented in child classes.")

    def forward(self, objective):
        """Compute gradients and apply an update rule to trainable variables.

        :parma objective: objective function, for which the gradients
            will be calculated
        :return: list of results
        """
        self.session.gradients(objective)
        apply_ops = [
            self.apply_gradient(x, grad=x.gradient)
            for x in self.trainable
        ]

        return self.session.run(*apply_ops)

    def backward(self, *args, **kwargs):
        """Return gradient of the operation by given inputs."""
        raise NoGradientException(
            f"There is no gradient for operation {self.name}."
        )

    def minimize(self, operation):
        """Set (target) operation for the optimizer."""
        wrapped_op = utils.convert_to_tensor(
            'constant',
            value=operation,
            name='optimizer-target-wrapper'
        )

        self.inputs = wrapped_op,
        return self


class BaseScaler:
    """Base scaler class."""

    def fit(self, *args, **kwargs):
        """Get scaler parameters from data sample."""
        raise NotImplementedError("Must be implemented in subclasses.")

    def transform(self, *args, **kwargs):
        """Transform data using scaler parameters."""
        raise NotImplementedError("Must be implemented in subclasses.")

    def fit_transform(self, *args, **kwargs):
        """Combine fit and transform methods."""
        raise NotImplementedError("Must be implemented in subclasses.")
