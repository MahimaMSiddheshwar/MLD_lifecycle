zenml init

zenml integration install sklearn xgboost lightgbm mlflow great_expectations evidently whylogs deepchecks -y

zenml experiment-tracker register mlflow_tracker --flavor=mlflow

zenml data-validator register ge_validator --flavor=great_expectations
zenml data-validator register deepchecks_validator --flavor=deepchecks
zenml data-validator register evidently_validator --flavor=evidently

zenml model-registry register mlflow_registry --flavor=mlflow

zenml stack register prod_stack \
  -a default \
  -o default \
  -d ge_validator \
  -d deepchecks_validator \
  -d evidently_validator \
  -e mlflow_tracker \
  -r mlflow_registry

zenml stack set prod_stack
