"""API v.1"""
from fastapi import FastAPI

from api import __version__
from api.v1.router import model_router

DESCRIPTION = """
Neural Network API🖧.

**Model served**: XOR

**Problem description**:

The XOR problem is a classic problem in neural networks researches.
The problem is this:

given the two inputs (data can only be '0' or '1'), we need to predict
the value of XOR function. Here are all the possible inputs and outputs:

| out | y   | x   |
|-----|-----|-----|
| 0   | 0   | 0   |
| 1   | 1   | 0   |
| 1   | 0   | 1   |
| 0   | 1   | 1   |

## Usage

You are be able to:
* **Predict data**
* **Get prediction results**
"""


app = FastAPI(
    title="NN-API",
    description=DESCRIPTION,
    version=__version__,
    contact={
        "name": "Corp.",
        "email": "nn-api@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://mit-license.org/",
    },
)


app.include_router(model_router)
