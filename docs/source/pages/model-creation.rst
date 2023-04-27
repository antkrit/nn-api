Create Neural Network model
===========================

.. toctree::
   :titlesonly:
   :maxdepth: 2

The model creation procedure always looks the same:

1. Initialize model with the expected shape of the input data

.. code-block:: python

    model = Model(input_shape=(1, 2))

2. Add some layers to the model

.. code-block:: python

    model.add(Dense(3))
    model.add(Dense(1))

3. Compile model.

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
parameter of the ``Model.fit()`` function). In most cases, your code
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

More examples can be found in `github repository <https://github.com/antkrit/nn-api/>`_:

- `MNIST Classifier <https://github.com/antkrit/nn-api/blob/main/notebooks/MNIST.example.ipynb>`_
- `SPAM Classifier <https://github.com/antkrit/nn-api/blob/main/notebooks/SPAM.example.ipynb>`_
- `XOR Solver <https://github.com/antkrit/nn-api/blob/main/notebooks/XOR.example.ipynb>`_
