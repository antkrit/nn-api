"""Contains global config object.

Expecting variables:
- LOG_LEVEL: logging level
- LOG_FORMAT: logging message format
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASEDIR = Path(__file__).resolve(strict=True).parent.parent.parent
CONFIG_PATH = BASEDIR / ".env"

load_dotenv(CONFIG_PATH)
settings = os.environ

LOG_LEVEL = settings.get("LOG_LEVEL", "DEBUG")
FORMAT = settings.get(
    "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "basic": {
            "()": "uvicorn.logging.DefaultFormatter",
            "format": FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "json": {
            "()": "api.utils.logging.JSONLogFormatter",
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "basic",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "json": {
            "formatter": "json",
            "class": "asynclog.AsyncLogDispatcher",
            "func": "api.utils.logging.write_log",
        },
    },
    "loggers": {
        "default": {
            "handlers": ["default"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "api": {
            "handlers": ["json"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "uvicorn": {
            "handlers": ["json"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": ["json"],
            "level": "ERROR",
            "propagate": False,
        },
    },
}
