"""API v.1"""
from fastapi import FastAPI

from api import __version__
from api.v1.router import model_router

DESCRIPTION = """
Neural Network API🖧.

**Model served**: MNIST Image Classifier

**Problem description**:

Dataset is a set of 70,000 28x28 images of digits handwritten
by high school students and employees of the US Census Bureau. Each image
is labeled with the digit it represents. The model must be able to recognize
the numbers in the image.

MNIST dataset: http://yann.lecun.com/exdb/mnist/

## Usage

You are be able to:
* **Predict data**
* **Get prediction results**
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


app.include_router(model_router)
