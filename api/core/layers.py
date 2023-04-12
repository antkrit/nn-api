"""Contains layers implementations."""
import abc
import functools

import numpy as np

from api.core import namespace
from api.core.autograd import Node, Session, ops
from api.core.autograd import utils as ag_utils
from api.core.preprocessing import initializers

__all__ = ("BaseLayer", "Dense", "Input", "InputShape")


class BaseLayer(metaclass=abc.ABCMeta):
    """Base layer class.

    In most cases, to create a custom layer, it is enough to
    override the `__init__()`, `build()` and `forward()` methods:
    - `build()` should be used to lazily add new layer variables
        and will be called before the first call of `self.forward()`
        method by `__call__()` method, in other words, this method was
        created to automatically calculate input shapes for new layer
        variables.
    - `forward()` should be used to implement layer logic, an algorithm
        that will calculate the layer's output.

    It is possible to override the `__call__()` method, but this is
    not recommended.

    So, a simple layer can be implemented as this:

    .. code-block:: python

        class Linear(BaseLayer):

            def __init__(self, units=10, name='Linear'):
                super().__init__(session=None, name=name)
                self.units = units

                self.weights = None
                self.bias = None

            def build(self, input_shape):
                self.weights = initializers.random_normal(
                    size=[input_shape[-1], self.units]
                )
                self.bias = initializers.ones(
                    size=[1, self.units]
                )
                self._built = True
                return input_shape[-2], self.units

            def forward(self, value, *args, **kwargs):
                return value @ self.weights + self.bias

    :param session: current session, if None, create a new one,
        defaults to None
    :param name: name of the layer, defaults to 'Layer'
    """

    # shape of the layer could be None
    # pylint: disable=inconsistent-return-statements

    def __init__(self, session=None, name="Layer"):
        """Constructor method."""
        self.session = session or Session()
        self.name = name

        self.input = None
        self.input_shape = None

        self._trainable = []
        self._non_trainable = []

        self._built = False

    @property
    def shape(self):
        """Return shape of the layer (without batch size)."""
        if self.input_shape:
            return self.input_shape.shape

    @property
    def batch(self):
        """Return batch size."""
        if self.input_shape:
            return self.input_shape.batch

    @property
    def batch_shape(self):
        """Return full shape of the data."""
        if self.input_shape:
            bshape = self.input_shape.batch_input_shape
            return tuple(x for x in bshape if x is not None)

    @property
    def built(self):
        """Access protected built argument."""
        return self._built

    @built.setter
    def built(self, _):
        msg = "Cannot be built manually. Instead, run the `build()` method."
        raise ValueError(msg)

    def build(self, *args, **kwargs):
        """Initialize layer variables.

        This method may be implemented by subclasses. It will be called
        before execution of `forward()`. Typically used to automatically
        define and validate I/O shapes.
        """
        del args, kwargs  # are used in subclasses
        self._built = True

    def add_variable(
        self, name=None, shape=None, initializer=None, trainable=False
    ):
        """Add new layer value.

        An initializer is expected to be of type class:`BaseInitializer`,
        which returns a variable. If None, use func:`xavier_uniform` as default

        New variables will be initialized with the name: layer.name/var_name.
        """
        var_name = name or "unknown-variable"
        initializer = initializer or initializers.xavier_uniform

        var = initializer(shape, name=self.name + "/" + var_name, shape=shape)
        if not trainable:
            self._non_trainable.append(var)
        else:
            self._trainable.append(var)

        return var

    def variables(self):
        """Return trainable variables."""
        return self._trainable

    @abc.abstractmethod
    def forward(self, x, *args, **kwargs):
        """Calculate output of the layer."""
        raise NotImplementedError("Must be implemented in subclasses.")

    def __call__(self, x, *args, **kwargs):
        """Perform forward step with some preprocessing.

        .. note::
            `input_shape` and `input_cache` keyword arguments is reserved for
            special purposes, see the function implementation for details.

        :param x: input data
        :return: layer output operation(result of `self.forward` function call)
        """
        if x is not None and not isinstance(x, Node):
            x = np.atleast_1d(x)

        # it is possible to pass input shape as keyword argument,
        # but it has the lowest priority and may be not used, if
        # none of the options are true, then input_shape will be
        # set to None
        input_shape = kwargs.pop("input_shape", None)

        # the first set input can be cached, but this
        # argument is ignored if `self.input` is None
        # or x is not None
        input_cache = kwargs.pop("input_cache", False)
        input_cache = (
            input_cache
            and hasattr(self, "input")
            and self.input is not None
            and x is None
        )

        if not input_cache:
            # input shape will be set during first function call,
            # or it can be set in a different way in custom layer
            # implementation (e.g. Input layer)
            if self.shape:
                input_shape = self.batch_shape
            elif x is not None and hasattr(x, "shape") and x.shape:
                input_shape = x.shape

            # if x is None, a placeholder will be created
            x = ag_utils.convert_to_node(x, shape=input_shape)

            input_shape = InputShape(input_shape)
            self.input = x
        else:
            x = self.input
            input_shape = self.input_shape

        self.build(input_shape.batch_input_shape)
        output = self.forward(x, *args, **kwargs)

        return output


class Dense(BaseLayer):
    """Dense (or Fully Connected) layer.

    The output of this layer is computed as *activate*(X @ W + b), where
    X is layer input, W - layer weights, b - layer bias, activate -
    some activation function. Bias and activate are optional and can
    be disabled when creating the layer.

    :param units: number of layer units, defaults to 1
    :param activation: activation function, if None - the activation function
        will not be applied to the resulting formula, defaults to None
    :param weight_initializer: str or callable, name of the function
        (or function itself) used to initialize the weights, if None -
        `xavier_uniform` will be used as default, defaults to None
    :param bias_initializer: str or callable, name of the function
        (or function itself) used to initialize the biases, if None -
        `zeros` will be used as default, defaults to None
    :param use_bias: whether to use a bias when the forward pass
    """

    def __init__(
        self,
        units=1,
        activation=None,
        weight_initializer=None,
        bias_initializer=None,
        use_bias=True,
        session=None,
        name="Dense",
        **kwargs,
    ):
        """Constructor method."""
        super().__init__(session=session, name=name)
        self.units = units

        if not isinstance(activation, str) or activation is None:
            self.activation = activation
        else:
            self.activation = namespace.activations(
                activation, compiled=True, session=session, **kwargs
            )

        self.weight_initializer = namespace.initializers(
            weight_initializer or namespace.initializers.xavier_uniform,
            compiled=False,
        )
        self.bias_initializer = namespace.initializers(
            bias_initializer or namespace.initializers.zeros, compiled=False
        )

        self.use_bias = use_bias

        # should be initialized with build() method
        self.weight = None
        self.bias = None

    def build(self, input_shape):
        if self._built:
            return

        if input_shape is None:
            raise ValueError("`input_shape` cannot be None.")

        if len(input_shape) < 1:
            raise ValueError(
                f"Dense layer input must have at least 1 dimension."
                f"Full input shape received: {input_shape}"
            )

        last_dim = input_shape[-1]

        input_shape = list(input_shape)
        input_shape[-2:] = (last_dim, self.units)
        self.input_shape = InputShape(input_shape)

        self.weight = self.add_variable(
            "weight",
            shape=(last_dim, self.units),
            initializer=self.weight_initializer,
            trainable=True,
        )

        if self.use_bias:
            self.bias = self.add_variable(
                "bias",
                shape=(self.units,),
                initializer=self.bias_initializer,
                trainable=True,
            )

        self._built = True

    def forward(self, value, *args, **kwargs):
        output = ops.einsum("bhw, wk -> bhk", value, self.weight, **kwargs)

        if self.use_bias:
            output = ops.add(output, self.bias, **kwargs)

        if self.activation:
            return self.activation(output, *args, **kwargs)

        return output


class Input(BaseLayer):
    """Input layer.

    By default, the input layer is required to know the size of the input
    data. When calling a layer without arguments, a Placeholder with the
    input_shape mentioned above will be created.

    If the Input layer was called with some argument x (in other words, the
    input data was passed directly) - it will return a Variable with that data

    :param input_shape: fixed shape of future data, at least 1d
    :raises ValueError: if input_shape dimension < 1
    """

    def __init__(
        self, input_shape, batch_size=None, session=None, name="Input"
    ):
        """Constructor method."""
        super().__init__(session=session, name=name)

        if not input_shape:
            msg = f"Input shape must be at least 1d, received: {input_shape}"
            raise ValueError(msg)

        batch_input_shape = (batch_size, *input_shape)
        self.input_shape = InputShape(batch_input_shape)

    def forward(self, x, *args, **kwargs):
        """Calculate output of the Input layer."""
        return x

    # by default for the Input layer, this method should receive
    # x == None as input, this is necessary so that the Placeholder
    # is automatically created during function call
    # this behavior can be changed manually by passing x argument
    # in this case, Variable will be created instead of Placeholder
    __call__ = functools.partialmethod(
        BaseLayer.__call__, x=None, input_cache=True
    )


class InputShape:
    """Contains information about the shape of the input data.

    If the length of the batch_input_shape is less than 2, then
    batch will be set to None. Otherwise, the first element of
    the batch_input_shape will be the batch size. The rest of
    the batch_input_shape elements are considered input_shape.

    :param batch_input_shape: array that contains at least 2 elements,
        the first element is batch size and all the rest are input shape.
    :raises ValueError: array length is less than 2
    """

    def __init__(self, batch_input_shape):
        """Constructor method."""
        if isinstance(batch_input_shape, list):
            batch_input_shape = tuple(batch_input_shape)

        # if batch_input_shape length is greater than or equal to 2,
        # then batch_idx will be equal to 1 and 0 otherwise
        batch_idx = int(len(batch_input_shape) >= 2)
        batch_size = batch_input_shape[0:batch_idx]

        self.batch = batch_size[0] if batch_size else None
        self.shape = batch_input_shape[batch_idx:]

        self.batch_input_shape = (self.batch, *self.shape)
