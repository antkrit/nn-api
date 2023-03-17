"""Contains layers implementations."""
import abc
import functools
import numpy as np

from api.core import namespace
from api.core.preprocessing import initializers
from api.core.autograd import Session, Node, utils as ag_utils


__all__ = ('BaseLayer', 'Dense', 'Input')


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

            def forward(self, value, *args, **kwargs):
                return value @ self.weights + self.bias

    :param session: current session, if None, create a new one,
        defaults to None
    :param name: name of the layer, defaults to 'Layer'
    """
    def __init__(self, session=None, name='Layer'):
        """Constructor method."""
        self.session = session or Session()
        self.name = name

        self._trainable = []
        self._non_trainable = []

        self._built = False

    def build(self, input_shape):
        """Initialize layer variables.

        This method may be implemented by subclasses. It will be called
        before execution of `forward()`. Typically used to automatically
        define and validate I/O shapes.
        """
        self._built = True

    def add_variable(
            self,
            name=None,
            shape=None,
            initializer=None,
            trainable=False
    ):
        """Add new layer value.

        An initializer is expected to be of type class:`BaseInitializer`,
        which returns a variable. If None, use func:`xavier_uniform` as default

        New variables will be initialized with the name: layer.name/var_name.
        """
        var_name = name or 'unknown-variable'
        initializer = initializer or initializers.xavier_uniform

        var = initializer(
            shape,
            name=self.name + '/' + var_name,
            shape=shape
        )
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
            `input_shape` keyword argument is reserved for special purposes,
            see the function implementation for details.

        :param x: input data
        :return: layer output operation(result of `self.forward` function call)
        """
        if x is not None and not isinstance(x, Node):
            x = np.atleast_1d(x)

        # input shape will be set during first function call,
        # or it can be set in a different way in custom layer
        # implementation (e.g. Input layer)
        if hasattr(self, 'input_shape'):
            input_shape = self.input_shape
        else:
            if x is not None and hasattr(x, 'shape'):
                input_shape = x.shape
            else:
                # it is possible to pass input shape as keyword argument,
                # but it has the lowest priority and may be not used,
                # if none of the options are true, then input_shape
                # will be set to None
                input_shape = kwargs.pop('input_shape', None)

        # if x is None, a placeholder will be created
        x = ag_utils.convert_to_node(x, shape=input_shape)

        self.build(input_shape)

        self.input = x
        self.input_shape = input_shape

        return self.forward(x, *args, **kwargs)


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
        xavier_uniform will be used as default, defaults to None
    :param bias_initializer: str or callable, name of the function
        (or function itself) used to initialize the biases, if None -
        zeros will be used as default, defaults to None
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
            name='Dense',
            *args, **kwargs
    ):
        """Constructor method."""
        super().__init__(session=session, name=name)
        self.units = units

        if isinstance(activation, str):
            self.activation = namespace.activations(
                activation,
                compiled=True,
                session=session,
                *args, **kwargs
            )
        else:
            self.activation = activation

        if isinstance(weight_initializer, str):
            self.weight_initializer = namespace.initializers(
                weight_initializer,
                compiled=False
            )
        else:
            self.weight_initializer = weight_initializer \
                                      or namespace.initializers.xavier_uniform

        if isinstance(bias_initializer, str):
            self.bias_initializer = namespace.initializers(
                bias_initializer,
                compiled=False
            )
        else:
            self.bias_initializer = bias_initializer \
                                    or namespace.initializers.zeros

        self.use_bias = use_bias

        # should be initialized with build() method
        self.weight = None
        self.bias = None

    def build(self, input_shape):
        if self._built:
            return

        if len(input_shape) < 1:
            raise ValueError(
                "Dense layer input must have at least 1 dimension."
                f"Full input shape received: {input_shape}"
            )

        last_dim = input_shape[-1]

        self.weight = self.add_variable(
            "weight",
            shape=[last_dim, self.units],
            initializer=self.weight_initializer,
            trainable=True
        )

        if self.use_bias:
            self.bias = self.add_variable(
                "bias",
                shape=[
                    self.units,
                ],
                initializer=self.bias_initializer,
                trainable=True
            )

        self._built = True

    def forward(self, value, *args, **kwargs):
        output = value @ self.weight
        output = output + self.bias if self.use_bias else output

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

    def __init__(self, input_shape, session=None, name='Input'):
        """Constructor method."""
        super().__init__(session=session, name=name)

        if not input_shape:
            msg = f"Input shape must be at least 1d, received: {input_shape}"
            raise ValueError(msg)

        self.input_shape = input_shape

    def forward(self, x, *args, **kwargs):
        """Calculate output of the Input layer."""
        return x

    # by default for the Input layer, this method should receive
    # x == None as input, this is necessary so that the Placeholder
    # is automatically created during function call
    # this behavior can be changed manually by passing x argument
    # in this case, Variable will be created instead of Placeholder
    __call__ = functools.partialmethod(BaseLayer.__call__, x=None)
