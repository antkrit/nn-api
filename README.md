# Neural Network API
[![Build and tests](https://github.com/antkrit/nn-api/actions/workflows/build.yml/badge.svg)](https://github.com/antkrit/nn-api/actions/workflows/build.yml)
[![Documentation Status](https://readthedocs.org/projects/nn-api/badge/?version=latest)](https://nn-api.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/antkrit/nn-api/branch/main/graph/badge.svg?token=WL1AOMBDYR)](https://codecov.io/gh/antkrit/nn-api)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)



## Description

The NN API is a simple API that serves the MNIST model for handwritten digit recognition.
The MNIST model is built and trained using an internal neural network library.

## Installation

### Prerequisites

Make sure you have installed all the following prerequisites on your development machine:
- Python 3.7+ (*with `setuptools`,  `wheel` and `virtualenv` packages*)
- Docker

## Set up project

- **Clone repository**
```bash
git clone https://github.com/antkrit/nn-api.git
```

- **Move into project root folder:**
```bash
cd nn-api
```

- **Create and activate virtual environment:**

*Linux:*
```bash
virtualenv venv
source venv/bin/activate
```

*Windows:*
```bash
virtualenv venv
venv\Scripts\activate
```

- **Install dependencies:**

Production requirements
(*if you are only interested in launching the application,
these requirements are enough for you*)
```bash
python -m pip install -e .
```

Development requirements (*includes production requirements*)

```bash

python -m pip install -e .[dev]

```

## Run application

### Model

The MNIST model training process can be found in the [MNIST example notebook](notebooks/MNIST.example.ipynb).
After training the model (and exporting it to a ".pkl"-like file), the path to it must be specified in the
[.env](.env) file



### Configuration

At least the following environment variables must be set (values may vary):

```dotenv
MODEL_PATH=api/model/trained_model-0.1.0-MNIST-DNN.pkl

CELERY_BROKER_URI=amqp://rabbitmq
CELERY_BACKEND_URI=redis://redis:6379/0
```

More specific project configuration, such as logging, can be found in the [config.py](api/v1/config.py) file.
Requests are logged in json format by default. This behavior can be changed by specifying the appropriate
logger for the [API logging middleware](api/v1/__init__.py). A list of pre-configured loggers can be found
in the [logging configuration](api/v1/config.py).



### Run development server

In the project root directory run:

```bash
docker compose build
docker compose up
```

After a successful build and launch, the following services will be available to you:
- http://localhost:8010/ - API
- http://localhost:8010/api/v1/docs - API v1 documentation (Swagger UI)
- http://localhost:5555/ - Celery task monitoring tool (Flower)�