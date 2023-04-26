"""Contains celery_service tasks implementation."""
import importlib
import logging
from abc import ABC

from celery import Task

from api.v1.worker import worker

logger = logging.getLogger("default")


class WrappedTask(Task, ABC):
    """A Celery `Task` class wrapper."""

    def __init__(self):
        """Constructor method."""
        super().__init__()
        self.model = None

    def __call__(self, *args, **kwargs):
        """Load and cache the model on the first call."""
        if not self.model:
            # additional args is defined by the `worker.task()` decorator
            # pylint: disable=no-member
            logger.info(
                "Import model. Wrapper path: %s.%s",
                self.model_module,
                self.model_object,
            )

            module_import = importlib.import_module(self.model_module)
            model_obj = getattr(module_import, self.model_object)

            self.model = model_obj()

            logger.info("Model imported.")

        return self.run(*args, **kwargs)


@worker.task(
    ignore_result=False,
    bind=True,
    base=WrappedTask,
    # make sure the `model_module` points to a file that has a model wrapper
    # implementation, and the `model_object` to the name of the wrapper
    model_module="api.model.wrapper",
    model_object="Model",
)
def predict_task(self, input_data):
    """Return model prediction."""

    if isinstance(input_data, dict):
        input_data = tuple(input_data.values())

    return self.model.predict(input_data).tolist()
