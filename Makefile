SHELL:=/bin/bash

env:
	python3 -m venv nbenv && source ./nbenv/bin/activate

clean:
	rm -rf ./nbenv && \
	rm -rf dist/ && \
	rm -rf build/ && \
	rm -rf nimminers.egg-info/ && \
	rm -rf .pytest_cache/ && \
	rm -rf .hypothesis

format:
	black --line-length 80 setup.py && \
	black  --line-length 80 ./src

lint:
	pylint *py

test:
	pytest tests/unit_tests/*

integration-test:
	pytest tests/integration_tests/*

