"""Contains model wrapper."""
from pathlib import Path

import joblib

from api import __version__

BASE_DIR = Path(__file__).resolve(strict=True).parent
MODEL_FILENAME = f"trained_model-{__version__}.pkl"
MODEL_PATH = BASE_DIR.joinpath(MODEL_FILENAME)


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
