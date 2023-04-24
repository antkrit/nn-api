import os

from api.core.generic import Model
from api.model.wrapper import Model as ModelWrapper


def test_wrapper(mocker):
    model = Model(input_shape=[1, 1])
    model.compile("gradient_descent", "mean_squared_error")

    mocker.patch("api.model.wrapper.Model._load_model", return_value=model)
    mocker.patch("api.core.generic.Model.predict", return_value="test")

    os.environ["MODEL_PATH"] = "_"

    model_wrapper = ModelWrapper()
    assert model_wrapper.predict([1, 1]) == model.predict([1, 1])
