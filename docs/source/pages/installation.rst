Quick Start
============

.. toctree::
   :maxdepth: 2

************
Installation
************

Prerequisites
#############

Make sure you have installed all the following prerequisites on your development machine:

- Python 3.7+ (*with `setuptools`,  `wheel` and `virtualenv` packages*)
- Docker

**************
Set up project
**************

- **Clone repository**

.. code-block:: bash

    git clone https://github.com/antkrit/nn-api.git

- **Move into project root folder:**

.. code-block:: bash

    cd nn-api

- **Create and activate virtual environment:**

*Linux:*

.. code-block:: bash

    virtualenv venv
    source venv/bin/activate

*Windows:*

.. code-block:: bash

    virtualenv venv
    venv\Scripts\activate

- **Install dependencies:**

Production requirements
(*if you are only interested in launching the application,
these requirements are enough for you*)

.. code-block:: bash

    python -m pip install -e .

Development requirements (*includes production requirements*)

.. code-block:: bash

    python -m pip install -e .[dev]

***************
Run application
***************

Model
#####

The MNIST model training process can be found in the
`MNIST example notebook <https://github.com/antkrit/nn-api/blob/main/notebooks/MNIST.example.ipynb>`_.
After training the model (and exporting it to a ".pkl"-like file), the path to it must be specified in the
`.env <https://github.com/antkrit/nn-api/blob/main/.env>`_ file.

Configuration
#############

At least the following environment variables must be set (values may vary):

.. code-block:: bash

    MODEL_PATH=api/model/trained_model-0.1.0-MNIST-DNN.pkl

    CELERY_BROKER_URI=amqp://rabbitmq
    CELERY_BACKEND_URI=redis://redis:6379/0

More specific project configuration, such as logging, can be found in the
`config.py <https://github.com/antkrit/nn-api/blob/main/api/v1/config.py>`_ file.
Requests are logged in json format by default. This behavior can be changed by specifying the appropriate
logger for the `API logging middleware <https://github.com/antkrit/nn-api/blob/dev/api/v1/__init__.py>`_.
A list of pre-configured loggers can be found in the
`logging configuration <https://github.com/antkrit/nn-api/blob/main/api/v1/config.py>`_.

Run development server
######################

In the project root directory run:

.. code-block:: bash

    docker compose build
    docker compose up

After a successful build and launch, the following services will be available to you:

- http://localhost:8010/ - API
- http://localhost:8010/api/v1/docs - API v1 documentation (Swagger UI)
- http://localhost:5555/ - Celery task monitoring tool (Flower)
