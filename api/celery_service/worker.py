"""Celery worker."""
import os

from celery import Celery

BROKER_URI = os.getenv("BROKER_URI")
BACKEND_URI = os.getenv("BACKEND_URI")

worker = Celery(
    "celery_app",
    broker=BROKER_URI,
    backend=BACKEND_URI,
    include=["celery_task_app.tasks"],
)

worker.conf.update(result_expires=3600)
