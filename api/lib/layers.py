"""Contains layers implementations."""
import api.lib.autograd as ag


__all__ = ('Dense',)


class BaseLayer:
    """Base layer class."""

    def forward(self, x, *args, **kwargs):
        """Calculate output of the layer."""
        raise NotImplementedError("Must be implemented in child classes.")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in child classes.")


class Dense(BaseLayer):
    """Linear (or Fully Connected) layer.

    :param size: input data size, defaults to (1, 1)
    :param activation: activation function, defaults to None
    :param weight_initializer: function used to initialize the weights and
        biases, defaults to None
    """
    def __init__(self, size=(1, 1), activation=None, weight_initializer=None):
        self.activation = activation
        self.weight_initializer = weight_initializer

        self.weights = self.weight_initializer(*size)
        self.bias = self.weight_initializer(1, *size[1:])

    def forward(self, x, *args, **kwargs):
        """Calculate output of the layer.

        :param x: input data
        """
        return self.activation(
            x @ self.weights + self.bias,
            *args, **kwargs
        )

    def __call__(self, x, *args, **kwargs):
        return ag.node_wrapper(self.forward, x, *args, **kwargs)
