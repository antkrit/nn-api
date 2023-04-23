"""Contains global config object."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASEDIR = Path(__file__).resolve(strict=True).parent.parent
CONFIG_PATH = BASEDIR / ".env"
MINIMAL_CONFIG = ["MODEL_PATH", "CELERY_BROKER_URI", "CELERY_BACKEND_URI"]


# use the f-string docstring to automatically update the `MINIMAL_CONFIG` list
# pylint: disable=missing-function-docstring
def load_config(path):
    load_dotenv(path)
    config = os.environ

    return config


settings = load_config(CONFIG_PATH)

load_config_docstring = f"""
    Load config from BASEDIR/.env file.

    .. note::
        The minimum configuration should contain at least {MINIMAL_CONFIG}
        env variables.

    :param path: path to `.env` file
    :raises ValueError: there are no required variables in the configuration
    :return: configuration dict
"""
load_config.__doc__ = load_config_docstring
