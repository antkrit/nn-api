"""Contains layers implementations."""
import api.lib.autograd as ag
import api.lib.namespace as namespace
from api.lib.bases import BaseLayer


class Dense(BaseLayer):
    """Linear (or Fully Connected) layer.

    :param size: input data size, defaults to (1, 1)
    :param activation: activation function, defaults to None
    :param weight_initializer: str, name of the function used to initialize
        the weights and biases, defaults to None
    """
    def __init__(
            self,
            size=(1, 1),
            activation=None,
            weight_initializer=None,
            session=None,
            *args, **kwargs
    ):
        """Constructor method."""
        super().__init__(session=session)

        self.size = size

        self.activation = namespace.activations(
            activation,
            compiled=True,
            session=session,
            *args, **kwargs
        )
        self.weight_initializer = namespace.initializers(
            weight_initializer,
            compiled=False
        )

        self.weights = self.weight_initializer(*size, *args, **kwargs)
        self.bias = self.weight_initializer(1, size[1], *args, **kwargs)
        self.trainable = [self.weights, self.bias]

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
