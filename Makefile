.PHONY: check install lint typecheck test format

install:
	uv sync --all-extras

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck:
	uv run mypy src/jevlab

test:
	uv run pytest -q

check: lint typecheck test
