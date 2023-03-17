import pytest
import numpy as np
from api.core.generic import Model
from api.core.layers import Dense
from api.core.loss import MSE
from api.core.activation import ReLU
from api.core.optimizers import GradientDescent
from api.core.exception import ModelIsNotCompiledException


def test_model_compilation():

    def _compare_trainable(left, right):
        if len(left) != len(right):
            return False

        try:
            for i, elem in enumerate(left):
                assert elem.name == right[i].name
        except AssertionError:
            return False
        return True

    model = Model()

    lr1 = Dense((1, 1), activation='relu', weight_initializer='ones')
    lr1_trainable = lr1.trainable
    lr2 = Dense((1, 1), activation='relu', weight_initializer='ones')
    lr2_trainable = lr2.trainable

    model.add(lr1)
    model.add(lr2)

    with pytest.raises(ValueError):
        model.add('lr1')

    all_trainable = lr1_trainable + lr2_trainable
    assert _compare_trainable(all_trainable, model.trainable)

    assert model.optimizer is None and model.loss is None

    opt = GradientDescent(learning_rate=1)
    model.compile(optimizer=opt, loss='mean_squared_error')

    assert isinstance(model.optimizer, GradientDescent)
    assert _compare_trainable(all_trainable, model.optimizer.trainable)
    assert isinstance(model.loss, MSE)

    test_lr = 1
    model.compile(
        optimizer='gradient_descent',
        loss='mean_squared_error',
        learning_rate=test_lr
    )

    assert isinstance(model.optimizer, GradientDescent)
    assert model.optimizer.learning_rate == test_lr
    assert _compare_trainable(all_trainable, model.optimizer.trainable)


@pytest.mark.parametrize('x', ([[2]], [[2, 2, 2]]), ids=['scalar', 'vector'])
def test_model_fit_predict(session, x):
    model = Model()

    output_shape = 1
    x = np.asarray(x)
    x_train = np.expand_dims(x, axis=0)
    y = np.ones(output_shape)

    layer = Dense(
        (x.size, output_shape),
        activation='relu',
        weight_initializer='ones'
    )
    w, b = layer.trainable
    model.add(layer)

    with pytest.raises(ModelIsNotCompiledException):
        model.fit(iter([x_train, y]), epochs=1)

    with pytest.raises(ModelIsNotCompiledException):
        model.predict(x)

    model.compile(
        optimizer='gradient_descent',
        loss='mean_absolute_error',
        learning_rate=1
    )

    # following tests assume:
    # learning rate = 1,
    # weight initialization set to 'ones',
    # relu activation is applied to positive values

    # 1 epoch: ReLU(X @ W + b)
    # b derivative = 1
    # W derivative = X.T
    model.fit(iter([[x_train, y]]), epochs=1)

    # after the optimization step w.value should be equal to w - x.T
    assert np.array_equal(
        w.value,
        np.ones(layer.size) - x.T
    )

    # after the optimization step b.value should be equal to 0
    # (if weight initializer is set to 'ones')
    assert np.array_equal(b.value, np.zeros((1, output_shape)))

    activation = ReLU()
    assert np.allclose(
        model.predict(x),
        session.run(activation(x.dot(w.value) + b.value))
    )
