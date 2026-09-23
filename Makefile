.PHONY: install test lint check phase01 phase02

install:
	python -m pip install -e '.[dev,tokens]'

test:
	PYTHONPATH=src pytest

lint:
	ruff check src tests

check: lint test

phase01:
	PYTHONPATH=src python -m agent_lab phase01

phase02:
	PYTHONPATH=src python -m agent_lab phase02 --text "EtherCAT distributed clocks and servo control"
