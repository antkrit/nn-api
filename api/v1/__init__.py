"""API v.1"""
import logging
from logging.config import dictConfig

from fastapi import FastAPI

from api import __version__
from api.v1.config import LOGGING_CONFIG
from api.v1.middleware import LoggingMiddleware
from api.v1.router import model_router

dictConfig(LOGGING_CONFIG)

DESCRIPTION = """
Neural Network API🖧.

**Model served**: MNIST Image Classifier

**Problem description**:

Dataset is a set of 70,000 28x28 images of digits handwritten
by high school students and employees of the US Census Bureau. Each image
is labeled with the digit it represents. The model must be able to recognize
the numbers in the image.

MNIST dataset: http://yann.lecun.com/exdb/mnist/
"""

app = FastAPI(
    title="NN-API",
    description=DESCRIPTION,
    version=__version__,
    license_info={
        "name": "MIT License",
        "url": "https://mit-license.org/",
    },
)

# Routes
app.include_router(model_router, prefix="/mnist", tags=["model"])

# Middlewares
app.add_middleware(LoggingMiddleware, logger=logging.getLogger(__name__))
