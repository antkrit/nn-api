"""Contains various base classes."""
import abc
import numpy as np
from api.lib.exception import NoGradientException
from api.lib.autograd import Session, Operation, Variable, Constant, ops


class BaseLayer:
    """Base layer class."""

    def __init__(self, session=None):
        """Constructor method."""
        self.trainable = []
        self.session = session or Session()

    @abc.abstractmethod
    def forward(self, x, *args, **kwargs):
        """Calculate output of the layer."""
        raise NotImplementedError("Must be implemented in child classes.")

    @abc.abstractmethod
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

    @abc.abstractmethod
    def forward(self, x):
        """Calculate function."""
        raise NotImplementedError("Must be implemented in child classes.")

    @abc.abstractmethod
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

    @abc.abstractmethod
    def forward(self, y_true, y_pred):
        """Calculate loss."""
        raise NotImplementedError("Must be implemented in child classes.")

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in child classes.")


class BaseInitializer:
    """Base Initializer class.

    :param seed: number in the range [0, 2**32], define the internal state of
        the generator so that random results can be reproduced, defaults to None
    """

    def __init__(self, seed):
        self.seed = seed

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in a subclass.")


class BaseOptimizer(Operation, metaclass=abc.ABCMeta):
    """Base optimizer class.

    :param session: current session, if None - creates new, defaults to None
    :param trainable_variables: variables to optimize, defaults to
        weight and bias
    """

    def __init__(self, trainable_variables, session=None, name=None, **kwargs):
        """Constructor method."""
        super().__init__(name=name, **kwargs)
        self.session = session or Session()
        self.trainable = trainable_variables
        self._variables = []
        self._iteration = None

    @abc.abstractmethod
    def build(self, var_list):
        """Initialize optimizer variables.

        This method should be implemented and called by subclasses.
        """
        if hasattr(self, '__built') and self.__built:
            return

        self._iteration = self.add_variable(0, 'optimizer-local-iteration')
        self._build_indexed_dict(var_list)

    def _build_indexed_dict(self, var_list):
        """Assign index to each trainable node."""
        self._index_dict = {}

        for i, v in enumerate(var_list):
            self._index_dict[v] = i

        return self._index_dict

    def add_variable(self, init_val, var_name=None):
        """Add new optimizer value.

        :param init_val: value of a variable to be created
        :param var_name: name of the node to be created
        :returns: an optimizer variable
        """
        var = Variable(value=init_val, name=var_name)
        self._variables.append(var)

        return var

    def add_variable_from_reference(
            self, model_var, var_name, init=None, shape=None
    ):
        """Create an optimizer variable from model variable.

        :param model_var: :class:`Variable` node. The corresponding model
            variable to the optimizer variable to be created.
        :param var_name: new optimizer value will be created with name
            'model_var.name/var_name'
        :param init: the initial value of the optimizer variable, if
            None, the initial value will be default to 0, defaults to None
        :param shape: shape of the optimizer variable value
        :returns: an optimizer variable
        """
        model_var_numpy = np.asarray(model_var.value)

        if init is None:
            init = np.zeros(model_var_numpy.shape)
        elif callable(init):
            if shape is None:
                shape = model_var_numpy.shape

            # init supposed to be of BaseInitializer type, which returns node
            init = init(shape).value

        var_name = f'{model_var.name}/{var_name}'

        var = Variable(value=init, name=var_name)
        self._variables.append(var)

        return var

    def variables(self):
        """Return all variables."""
        return self._variables

    @abc.abstractmethod
    def apply_gradient(self, *args, **kwargs):
        """Apply computed gradients to trainable variables."""
        raise NotImplementedError("Must be implemented in child classes.")

    def forward(self, objective):
        """Compute gradients and apply an update rule to trainable variables.

        :parma objective: objective function, for which the gradients
            will be calculated
        :return: list of results
        """
        self.build(self.trainable)
        self.session.gradients(objective)
        apply_ops = [
            self.apply_gradient(x, grad=Variable(x.gradient))
            for x in self.trainable
        ]
        iteration = ops.assign_add(self._iteration, 1)

        return self.session.run(*apply_ops, iteration, returns=apply_ops)

    def backward(self, *args, **kwargs):
        """Return gradient of the operation by given inputs."""
        raise NoGradientException(
            f"There is no gradient for operation {self.name}."
        )

    def minimize(self, operation):
        """Set (target) operation for the optimizer."""
        wrapped_op = Constant(value=operation, name='optimizer-target-wrapper')

        self.inputs = wrapped_op,
        return self


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
