#!/usr/bin/env bash

set -Eeo pipefail

poetry run zenml integration install sklearn xgboost lightgbm mlflow great_expectations evidently whylogs  -y

poetry run zenml data-validator register ge_validator --flavor=great_expectations
poetry run zenml data-validator register evidently_validator --flavor=evidently

poetry run zenml experiment-tracker register local_mlflow_tracker  --flavor=mlflow
poetry run zenml model-deployer register local_mlflow_deployer  --flavor=mlflow

poetry run zenml stack register local_stack \
    -a default \
    -o default \
    -e local_mlflow_tracker \
    -d local_mlflow_deployer \
    -dv ge_validator \
    -dv evidently_validator \
    -c sklearn \
    -c xgboost \
    -c lightgbm \
    -c whylogs

poetry run zenml stack set local_stack
