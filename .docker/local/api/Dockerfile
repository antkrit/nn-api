FROM       python:3.10-slim as base

ENV        PYTHONFAULTHANDLER=1 \
           PYTHONHASHSEED=random \
           PYTHONUNBUFFERED=1

COPY       ./.docker/local/api/entrypoint.sh /entrypoint
RUN        sed -i 's/\r$//g' /entrypoint
RUN        chmod +x /entrypoint

COPY       ./.docker/local/api/start.sh /start
RUN        sed -i 's/\r$//g' /start
RUN        chmod +x /start

COPY       ./.docker/local/api/celery/worker/start.sh /start-celeryworker
RUN        sed -i 's/\r$//g' /start-celeryworker
RUN        chmod +x /start-celeryworker

COPY       ./.docker/local/api/celery/flower/start.sh /start-flower
RUN        sed -i 's/\r$//g' /start-flower
RUN        chmod +x /start-flower

WORKDIR    /app
COPY       api/ api
COPY       setup.py .
COPY       setup.cfg .

FROM       base as builder

ENV        PIP_DEFAULT_TIMEOUT=100 \
           PIP_DISABLE_PIP_VERSION_CHECK=1

RUN        pip install --no-cache-dir -e .

ENTRYPOINT ["/entrypoint"]
