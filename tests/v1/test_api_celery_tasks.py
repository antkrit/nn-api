import numpy as np
import pytest

from api.v1.tasks import predict_task


def test_predict_task(load_custom_model_sgd_mse):
    model = load_custom_model_sgd_mse

    with pytest.raises(ValueError):
        data = np.ones((1, 1, 1, 4))
        assert data.shape != model.shape
        assert data.shape != model.batch_shape

        _ = predict_task.apply(args=(data,)).get()

    data = np.ones(model.shape)
    output = np.asarray(predict_task.apply(args=(data,)).get())
    assert output.shape == (1, 1, 5)
