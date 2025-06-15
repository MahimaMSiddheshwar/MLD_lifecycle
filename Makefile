.PHONY: install train test tune fmt lint

install:
	poetry install

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
