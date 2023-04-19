"""Contains model wrapper."""
import os

import joblib

MODEL_PATH = os.environ["MODEL_PATH"]


class Model:
    """:class:`api.core.generic.Model` wrapper."""

    def __init__(self):
        """Constructor method."""
        self.model = self._load_model(MODEL_PATH)

    @staticmethod
    def _load_model(path):
        """Load model with `joblib` module."""
        model = joblib.load(path)
        return model

    def predict(self, data, *args, **kwargs):
        """Model `predict()` method wrapper."""
        return self.model.predict([data], *args, **kwargs)
