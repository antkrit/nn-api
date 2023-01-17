"""Contains Model definition."""
# TODO: detailed docstrings
from api.lib.layers import Layer
from api.lib.autograd import Graph, Session, Placeholder


class Model:
    """Create a model that contains all the layers and has
     an interface to work with them.
     """

    def __init__(self, layers=None, loss=None, optimizer=None):
        """Constructor method
        """
        self.layers = layers or []
        self.loss = loss
        self.optimizer = optimizer

    def add(self, layer):
        """Add layer to the model."""
        if not isinstance(layer, Layer):
            raise ValueError("Object must be of `Layer` type.")
        self.layers.append(layer)

    def predict(self, data):
        """Predict output with given input."""
        samples = len(data)
        result = []

        Graph().as_default()
        X = Placeholder('x')

        output = X
        for layer in self.layers:
            output = layer(output)

        sess = Session()
        for i in range(samples):
            output = sess.run(output, {'x': data[i]})
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, *args, **kwargs):
        """Train model."""
        samples = len(x_train)

        Graph().as_default()
        X = Placeholder('x')
        y = Placeholder('y')

        output = X
        for layer in self.layers:
            output = layer.forward(output)
        loss_op = self.loss(output, y)

        sess = Session()
        optimizer_op = self.optimizer(*args, **kwargs).minimize(loss_op)

        for i in range(epochs):
            err = 0
            for j in range(samples):
                feed_dict = dict(x=x_train[j], y=y_train[j])

                err += sess.run(loss_op, feed_dict)
                sess.run(optimizer_op, feed_dict)

            err /= samples
            print(f"epoch: {i+1}/{epochs}, loss: {err}")
