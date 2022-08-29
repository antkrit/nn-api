"""Contains implementation of base layer class"""


class Layer:
    """Base layer class."""
    def __init__(self):
        """Constructor method
        """
        self.input_ = None
        self.output = None

    def forward(self, input_, *args, **kwargs):
        """Calculate output of the layer.

        :param input_: input data
        """
        raise NotImplementedError("Must be implemented in child classes.")
