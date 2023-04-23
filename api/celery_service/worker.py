"""Celery worker."""
from celery import Celery

from api.config import settings

BROKER_URI = settings.get("CELERY_BROKER_URI", "redis://localhost")
BACKEND_URI = settings.get("CELERY_BACKEND_URI", "redis://localhost")

worker = Celery(
    "celery_app",
    broker=BROKER_URI,
    backend=BACKEND_URI,
    include=["api.celery_service.tasks"],
)

worker.conf.update(result_expires=3600)
