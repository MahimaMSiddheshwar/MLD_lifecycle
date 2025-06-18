.PHONY: install train test tune fmt lint

env:
	poetry env use MLD_Lifecycle

install:
	poetry install
	zenml integration install s3 sklearn mlflow deepchecks -y

train:
	poetry run train

test:
	poetry run test

tune:
	poetry run tune

fmt:
	poetry run black .

lint:
	poetry run flake8 src/
