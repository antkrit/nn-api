#!/bin/bash

# if any of the commands in code fails for any reason, the entire script fails
set -o errexit
# exits if any of variables is not set
set -o nounset


worker_ready() {
    celery -A api.celery_service.worker inspect ping
}

until worker_ready; do
  >&2 echo 'Celery workers not available'
  sleep 1
done
>&2 echo 'Celery workers is available'

celery --broker ${CELERY_BROKER_URI} --result-backend ${CELERY_BACKEND_URI} -A api.celery_service.worker flower
