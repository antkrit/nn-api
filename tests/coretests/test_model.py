import pytest

from api.core.autograd import Placeholder
from api.core.exception import ModelIsNotCompiledException
from api.core.generic import Model
from api.core.layers import Dense
from api.core.loss import MSE
from api.core.optimizers import GradientDescent


@pytest.mark.parametrize(
    "optimizer",
    [GradientDescent(0.1), "gradient_descent"],
    ids=["optimizer=compiled", "optimizer=str"],
)
@pytest.mark.parametrize(
    "loss", [MSE(), "mean_squared_error"], ids=["loss=compiled", "loss=str"]
)
@pytest.mark.parametrize(
    "metrics",
    [(MSE(),), ("mean_squared_error",), ("mean_squared_error", MSE())],
    ids=["metrics=(compiled,)", "metrics=(str,)", "metrics=(str, compiled)"],
)
def test_model_creation(optimizer, loss, metrics):
    def _compare_trainable(left, right):
        if len(left) != len(right):
            return False

        try:
            for i, elem in enumerate(left):
                assert elem.name == right[i].name
        except AssertionError:
            return False
        return True

    model = Model(input_shape=(1, 1))

    lr1 = Dense(1, activation="relu", weight_initializer="ones")
    lr1_trainable = lr1.variables()
    lr2 = Dense(1, activation="relu", weight_initializer="ones")
    lr2_trainable = lr2.variables()

    model.add(lr1)
    assert model.output_shape == lr1.batch_shape

    model.add(lr2)
    assert model.output_shape == lr2.batch_shape

    with pytest.raises(ValueError):
        model.add("lr1")

    all_trainable = lr1_trainable + lr2_trainable
    assert _compare_trainable(all_trainable, model.variables())

    with pytest.raises(ModelIsNotCompiledException):
        model.fit(None, None)

    with pytest.raises(ModelIsNotCompiledException):
        model.predict(None)

    assert not model._built
    model.build()
    assert not model._built

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    assert isinstance(model.optimizer, GradientDescent)
    assert _compare_trainable(all_trainable, model.optimizer.trainable)
    assert isinstance(model.loss, MSE)
    assert all([isinstance(metric, MSE) for metric in model.metrics])


def test_model_forward():
    m = Model(input_shape=(1, 2))
    assert isinstance(m(), Placeholder)

    assert isinstance(m(), Placeholder)

    layer = Dense(1)
    m.add(layer=layer)
    assert not isinstance(m(), Placeholder)
