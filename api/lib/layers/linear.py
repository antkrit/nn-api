"""Contains implementation of Linear layer"""
from api.lib.layers.base import Layer


class Linear(Layer):
    """Linear (or Fully Connected) layer.

    :param size: input data size, defaults to (1, 1)
    :param activation_function: activation function, defaults to None
    :param weight_initializer: function used to initialize the weights and
        biases, defaults to None
    """
    def __init__(self,
                 size=(1, 1),
                 activation_function=None,
                 weight_initializer=None):
        super().__init__()

        self.activation = activation_function

        self.weights = weight_initializer(size)
        self.bias = weight_initializer((1, *size[1:]))

    def forward(self, data, *args, **kwargs):
        """Calculate output of the layer.

        :param data: input data
        """
        self.input_ = data
        self.output = self.activation(
            data @ self.weights + self.bias,
            *args, **kwargs
        )

        return self.output
