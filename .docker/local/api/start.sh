#!/bin/bash

# if any of the commands in code fails for any reason, the entire script fails
set -o errexit
# fail exit if one of pipe command fails
set -o pipefail
# exits if any of variables is not set
set -o nounset

uvicorn api.main:app --reload --reload-dir api --host 0.0.0.0
