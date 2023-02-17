"""Contains Model definition."""
# TODO: detailed docstrings
from functools import wraps
from api.lib import namespace
from api.lib.autograd import Session, Placeholder, utils


class Model:
    """Create a model that contains all the layers and has
     an interface to work with them.
     """

    def __init__(self):
        """Constructor method."""
        self.loss = None
        self.optimizer = None
        self.session = Session()
        self.trainable = []

        self.__output_op = None
        self.__x = Placeholder(name='x')
        self.__y = Placeholder(name='y')

    def is_compiled(self):
        """Check if model is compiled."""
        return self.loss is not None and \
               self.optimizer is not None

    def _control_compile(func):
        """Check for decorated methods if the model is compiled.

        Use this decorator for methods that require the model to be compiled.
        """
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.is_compiled():
                raise namespace.exceptions.ModelIsNotCompiled(
                    "Model must be compiled first."
                )
            return func(self, *args, **kwargs)

        return wrapper

    def add(self, layer):
        """Add layer to the model."""
        if not isinstance(layer, namespace.bases.Layer):
            raise ValueError("Object must be of `BaseLayer` type.")

        self.trainable.extend(layer.trainable)

        if self.__output_op is None:
            self.__output_op = layer(self.__x)
            return
        self.__output_op = layer(self.__output_op)

    @_control_compile
    def predict(self, data):
        """Predict output with given input."""
        samples = len(data)
        result = []

        for i in range(samples):
            feed_dict = utils.form_feed_dict([data[i]], self.__x)
            output = self.session.run(self.__output_op, feed_dict)
            result.append(output)

        return result

    @_control_compile
    def fit(self, x_train, y_train, epochs=100, max_step=1):
        """Train model."""
        J = self.loss(self.__output_op, self.__y)

        samples = len(x_train)
        for i in range(epochs):
            err = 0
            for j in range(samples):
                feed_dict = utils.form_feed_dict(
                    [[x_train[j]], [y_train[j]]],
                    self.__x, self.__y
                )

                err += self.session.run(J, feed_dict)
                self.optimizer.minimize(J)

            err /= samples
            if i % max_step == 0:
                print(f"epoch: {i}/{epochs}, loss: {err}")

    def compile(self, optimizer, loss, *args, **kwargs):
        """Compile model using given parameters."""
        if isinstance(optimizer, namespace.bases.Optimizer):
            self.optimizer = optimizer
            self.optimizer.trainable = self.trainable
        else:
            self.optimizer = namespace.optimizers(
                optimizer,
                compiled=True,
                trainable_variables=self.trainable,
                *args, **kwargs
            )

        self.loss = namespace.losses(loss, compiled=True, session=self.session)

    _control_compile = staticmethod(_control_compile)
