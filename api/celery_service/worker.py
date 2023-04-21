"""Celery worker."""
from celery import Celery

from api.config import config

BROKER_URI = config.get("CELERY_BROKER_URI", "pyamqp://guest@localhost//")
BACKEND_URI = config.get("CELERY_BACKEND_URI", "redis://localhost")

worker = Celery(
    "celery_app",
    broker=BROKER_URI,
    backend=BACKEND_URI,
    include=["api.celery_service.tasks"],
)

worker.conf.update(result_expires=3600)
