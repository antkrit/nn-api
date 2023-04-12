"""Contains `Model` definition."""
import functools
import numpy as np

# tqdm.auto is using to work in both a terminal and notebooks
from tqdm.auto import trange

from api.core import namespace, data as data_adapter
from api.core.autograd import utils as ag_utils
from api.core.layers import BaseLayer, Input

from api.core.model.utils import control_compile
from api.core.preprocessing.samplers import train_test_split


class Model(Input):
    """Neural network model.

    At its core, the model is an `Input` layer with train/predict features
    and some modifications (see `Model.forward()` implementation).

    The model creation procedure always looks the same:
    1. Initialize model with the expected shape of the input data

    .. code-block:: python

        model = Model(input_shape=(1, 2))

    2. Add some layers to the model

    .. code-block:: python

        model.add(Dense(3))
        model.add(Dense(1))

    3. Model compilation.

    .. warning::
        Without this step, the model will not be able to train and predict.

    .. code-block:: python

        model.compile(
            optimizer='gradient_descent',
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )

    Now, the model is ready for training and prediction. For training,
    two options for printing results are possible (see the `verbosity`
    parameter of the `Model.fit()` function). In most cases, your code
    will look something like this:

    .. code-block:: python
        model = Model(input_shape=(1, 2))

        model.add(Dense(3, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            optimizer='gradient_descent',
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )

        x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
        y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

        model.fit(
            x_train, y_train,
            validation_data=[x_train, y_train],
            batch_size=4,
            epochs=10000,
            verbosity=1  # progress bar
        )

    For prediction, two options are possible:
    - Pass only the input data (x_test) to the function. In this case,
    the function will return only the predicted data.

    .. code-block:: python

        model.predict(x_test)

    - Pass both x_test and y_test data. In this case, the function will return
    the predicted data and print the estimated metrics for that data

    .. code-block:: python

        model.predict(x_test, y_test)

    :param input_shape: tuple, input data shape (without batch size)
    :param session: current session, if None - creates new, defaults to None
    :param name: model name, defaults to 'model'
    """

    def __init__(self, input_shape, session=None, name='model'):
        """Constructor method."""
        super().__init__(
            input_shape=input_shape,
            session=session,
            name=name
        )

        self.output = None
        self.output_shape = self.batch_shape

        self.loss = None
        self.metrics = []
        self.metric_names = []
        self.optimizer = None

        # predefined progress bar style for easier use in code
        self.__progress_bar = functools.partial(trange, ascii=True)

    def forward(self, x, *args, **kwargs):
        # return x if no layers have been added to the model yet
        # (which means `self.output` is None) or `self.output`
        return self.output or x

    def add(self, layer):
        """Add layer to the model.

        :param layer: BaseLayer instance, layer to add
        :raises ValueError: if layer is not of BaseLayer instance
        """
        if not isinstance(layer, BaseLayer):
            raise ValueError("Object must be of `BaseLayer` type.")

        self.output = layer(self(), input_shape=self.output_shape)
        self.output_shape = layer.batch_shape
        self._trainable.extend(layer.variables())

    @control_compile
    def fit(
            self,
            x_train,
            y_train,
            validation_data=None,
            validation_split=0.0,
            batch_size=1,
            shuffle=False,
            epochs=1,
            verbosity=1
    ):
        """Train model for a fixed number of epochs.

        .. note::
            To enable cross-validation, just pass validation_data or
            validation_split arguments. In this case, the `validation_data`
            argument will have priority.

        :param x_train: input data
        :param y_train: target (true) outputs
        :param validation_data: array with length 2 - [x_val, y_val],
            defaults to None
        :param validation_split: float in range (0, 1), the proportion of
            the train dataset to include in the validation split. For example,
            if validation split = 0.25, then the validation data will be 25%
            of the training data, defaults to 0.0
        :param batch_size: number of samples per gradient update, defaults to 1
        :param shuffle: whether to shuffle the data on epoch end,
            defaults to False
        :param epochs: number of epochs to train model, defaults to 1
        :param verbosity: 0 or 1, verbosity mode. 0 - silent mode,
            1 - progress bar, defaults to 1
        :raises ModelIsNotCompiledException: if this method was called without
            first compiling the model
        """
        if validation_data is None:
            x_val, x_train, y_val, y_train = train_test_split(
                x_train,
                y_train,
                split=validation_split
            )
        else:
            x_val, y_val = validation_data

        dataset_train = data_adapter.Dataset(
            x_train, y_train,
            dim=self.shape,
            batch_size=batch_size,
            shuffle=shuffle
        )
        dataset_validation = data_adapter.Dataset(
            x_val, y_val,
            dim=self.shape,
            batch_size=len(x_val),
            shuffle=shuffle
        )

        epochs = self.__progress_bar(epochs) if verbosity else range(epochs)

        for _ in epochs:

            # the dataset is infinite and has a limited number (its length)
            # of unique batches, so the loop iterates not over the dataset,
            # but over its length
            for _ in range(len(dataset_train)):
                # perform a learning step for each batch
                self.train_step(next(dataset_train))

            val_score = self.compute_metrics(next(dataset_validation))
            if val_score is None:
                # if `compute_metrics` returns None, then there
                # is no validation data. In this case, print only
                # `self.loss` value
                val_score = {self.loss.name: self.loss.value}

            if verbosity:
                epochs.set_postfix(val_score)

    @control_compile
    def predict(self, x_test, y_test=None):
        """Predict output with given input.

        :param x_test: input data
        :param y_test: target output. if the target data was passed
            to the function, not only will the predicted data be returned,
            but also the calculated metrics will be output to the console,
            defaults to None
        :raises ModelIsNotCompiledException: if this method was called without
            first compiling the model
        :return: predicted output
        """
        dataset_test = data_adapter.Dataset(
            x_test, np.empty((len(x_test),)) if y_test is None else y_test,
            dim=self.shape,
            batch_size=len(x_test),
            shuffle=False
        )

        output = None
        epochs = self.__progress_bar(1)

        for _ in epochs:
            output = self.predict_step(next(dataset_test))
            if y_test is not None:
                epochs.set_postfix(self.compute_metrics(next(dataset_test)))

        return output

    def compile(self, optimizer=None, loss=None, metrics=None):
        """Compile model using given parameters.

        :param optimizer: str or callable, model optimizer,
            defaults to `gradient_descent`
        :param loss: str or callable, model loss, defaults
            to `mean_squared_error`
        :param metrics: array of str or callable, additional metrics
            to evaluate, defaults to None
        """
        optimizer = optimizer or 'gradient_descent'
        loss = loss or 'mean_squared_error'

        self.optimizer = namespace.optimizers(optimizer, compiled=True)
        self.optimizer.session = self.session
        self.optimizer.trainable = self.variables()

        self.loss = namespace.losses(
            loss,
            name='loss',
            compiled=True,
            session=self.session
        )

        if metrics:
            self.metrics = [
                namespace.losses(metric, compiled=True, session=self.session)
                for metric in metrics
            ]

        self.metric_names = tuple(
            metric.name
            for metric in (self.loss, *self.metrics)
        )

        self._built = True

    def build(self, *args, **kwargs):
        # override the parent method so that the `_built` argument
        # can only be changed with the `compile()` method
        return

    def compute_metrics(self, data):
        """Compute metrics for given data.

        :param data: input data
        :return: dict with pairs metric.name:metric.value
        """
        x, y = data_adapter.unpack_x_y(data)

        # The len(array) == 0 construct is used because of
        # numpy's DeprecationWarning: The true value of an
        # empty array is ambiguous.
        if len(x) == 0 or len(y) == 0:
            return None

        y_pred = self()
        loss = self.loss(y_pred, y)
        metrics = [metric(y_pred, y) for metric in self.metrics]

        feed_dict = ag_utils.form_feed_dict([x], self.input)

        results = np.atleast_1d(
            self.session.run(loss, *metrics, feed_dict=feed_dict)
        )

        return dict(zip(self.metric_names, results))

    def train_step(self, data):
        """Perform one training step."""
        x, y = data_adapter.unpack_x_y(data)

        y_pred = self()
        loss = self.loss(y_pred, y)
        optimize = self.optimizer.minimize(loss)

        feed_dict = ag_utils.form_feed_dict([x], self.input)

        return self.session.run(
            loss,
            optimize,
            returns=[loss],
            feed_dict=feed_dict
        )

    def predict_step(self, data):
        """Perform one predict step."""
        x, _ = data_adapter.unpack_x_y(data)
        feed_dict = ag_utils.form_feed_dict([x], self.input)

        return self.session.run(self(), feed_dict=feed_dict)
