"""Contains Model definition."""
# TODO: detailed docstrings
import numpy as np
from api.lib import namespace
from api.lib.autograd import Session, utils as ag_utils
from api.lib.model.utils import control_compile


class Model:
    """Create a model that contains all the layers and has
     an interface to work with them.
     """

    def __init__(self):
        """Constructor method."""
        self.loss = None
        self.optimizer = None
        self.trainable = []

        self.__output_op = None
        self.__input = ag_utils.convert_to_tensor('placeholder', name='x')
        self.__y_true = ag_utils.convert_to_tensor('placeholder', name='f(x)')

        self.session = Session()

    def is_compiled(self):
        """Check if model is compiled."""
        return self.loss is not None and \
               self.optimizer is not None

    def add(self, layer):
        """Add layer to the model."""
        if not isinstance(layer, namespace.bases.Layer):
            raise ValueError("Object must be of `BaseLayer` type.")

        self.trainable.extend(layer.trainable)

        if self.__output_op is None:
            self.__output_op = layer(self.__input)
            return
        self.__output_op = layer(self.__output_op)

    @control_compile
    def predict(self, data):
        """Predict output with given input."""
        return self.session.run(
            self.__output_op,
            ag_utils.form_feed_dict(data, self.__input)
        )

    @control_compile
    def fit(self, train, epochs=100, max_step=1):
        """Train model."""
        cost = self.loss(self.__output_op, self.__y_true)
        optimize = self.optimizer.minimize(cost)

        for i in range(epochs):
            feed_dict = ag_utils.form_feed_dict(
                next(train),
                self.__input, self.__y_true
            )

            err, *_ = self.session.run([cost, optimize], feed_dict)

            err = np.mean(err)
            if i % max_step == 0:
                print(f"epoch: {i}/{epochs}, err: {err}")

    def compile(self, optimizer, loss, *args, **kwargs):
        """Compile model using given parameters."""
        if isinstance(optimizer, namespace.bases.Optimizer):
            self.optimizer = optimizer
            self.optimizer.session = self.session
            self.optimizer.trainable = self.trainable
        else:
            self.optimizer = namespace.optimizers(
                optimizer,
                compiled=True,
                trainable_variables=self.trainable,
                session=self.session,
                *args, **kwargs
            )

        self.loss = namespace.losses(loss, compiled=True, session=self.session)
