from celery.result import AsyncResult

from api.v1.tasks import predict_task
from tests.utils import create_in_memory_image


def test_mnist_predict(mocker, client):
    mocker.patch.object(predict_task, "delay", return_value=1)

    filetype = "gif"
    filename = f"gif.{filetype}"
    file = create_in_memory_image(
        size=(10, 10), filename=filename, filetype=filetype
    )

    response = client.post(
        "api/v1/mnist/predict", files={"file": (filename, file, "image/jpeg")}
    )
    assert response.status_code == 415

    filetype = "jpeg"
    filename = f"image.{filetype}"
    file = create_in_memory_image(
        size=(28, 28), filename=filename, filetype=filetype
    )

    response = client.post(
        "api/v1/mnist/predict", files={"file": (filename, file, "image/jpeg")}
    )
    assert response.status_code == 202


def test_mnist_task_id(mocker, client):
    max_value = 0.9
    max_value_id = 1
    fake_task_result = [1 - max_value, max_value]

    class FakeTask:
        def __init__(self, status, ready, result):
            self.status = status
            self.result = result
            self.is_ready = ready

        def ready(self):
            return self.is_ready

        def __call__(self, *args, **kwargs):
            return self

    mocker.patch.object(
        AsyncResult,
        "__new__",
        return_value=FakeTask(
            status="PROCESSING", ready=False, result=fake_task_result
        ),
    )

    response = client.get("api/v1/mnist/some_id?limit=1")
    assert response.status_code == 202

    mocker.patch.object(
        AsyncResult,
        "__new__",
        return_value=FakeTask(
            status="SUCCESS", ready=True, result=fake_task_result
        ),
    )

    response = client.get("api/v1/mnist/some_id?limit=1")
    assert response.status_code == 200

    result = response.json()["result"]
    # example response:
    # {..., 'result': [{'class': 1, 'probability': '80.00%'}]}
    assert len(result) == 1  # equal to ?limit=
    assert result[0]["class"] == max_value_id

    response = client.get("api/v1/mnist/some_id?limit=5")
    assert response.status_code == 200

    result = response.json()["result"]
    # larger of the two values: {len(fake_task_result); ?limit=}
    assert len(result) == len(fake_task_result)
    assert result[0]["class"] == max_value_id
