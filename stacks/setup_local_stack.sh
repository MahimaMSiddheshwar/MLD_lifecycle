#!/usr/bin/env bash

set -Eeo pipefail

zenml integration install sklearn xgboost lightgbm mlflow great_expectations evidently whylogs deepchecks -y

zenml data-validator register ge_validator --flavor=great_expectations
zenml data-validator register deepchecks_validator --flavor=deepchecks
zenml data-validator register evidently_validator --flavor=evidently

zenml experiment-tracker register local_mlflow_tracker  --flavor=mlflow
zenml model-deployer register local_mlflow_deployer  --flavor=mlflow

zenml stack register local_gitflow_stack \
    -a default \
    -o default \
    -e local_mlflow_tracker \
    -d local_mlflow_deployer \
    -dv deepchecks_data_validator

zenml stack set local_gitflow_stack
