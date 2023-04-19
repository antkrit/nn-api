"""Contains celery_service tasks implementation."""
import importlib
from abc import ABC

from celery import Task

from api.celery_service.worker import worker


class WrappedTask(Task, ABC):
    """A Celery `Task` class wrapper."""

    def __init__(self):
        """Constructor method."""
        super().__init__()
        self.model = None

    def __call__(self, *args, **kwargs):
        """Load and cache the model on the first call."""
        if not self.model:
            # `self.path` argument is defined by the `worker.task()` decorator
            # pylint: disable=no-member
            module_import = importlib.import_module(self.path[0])
            model_obj = getattr(module_import, self.path[1])
            self.model = model_obj()

        return self.run(*args, **kwargs)


# Important!
# make sure the first element in the `path` array points to a
# file that has a model wrapper implementation, and the second
# to the name of the wrapper
@worker.task(
    ignore_result=False,
    bind=True,
    base=WrappedTask,
    path=("api.wrapper", "Model"),
    name=f"{__name__}.{'Model'}",
)
def predict_task(self, input_data):
    """Return model prediction."""
    if isinstance(input_data, dict):
        input_data = tuple(input_data.values())

    return self.model.predict(input_data)
