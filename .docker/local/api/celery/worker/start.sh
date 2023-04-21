#!/bin/bash

# if any of the commands in code fails for any reason, the entire script fails
set -o errexit
# exits if any of variables is not set
set -o nounset

celery -A api.celery_service.worker worker --loglevel=info
