.PHONY: install dev lint typecheck test test-cov test-all docker-build docker-up run clean

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

install:
	uv sync

dev:
	uv sync --extra dev

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

lint-fix:
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v

test-property:
	uv run pytest tests/property/ -v

test-all:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

run:
	uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker-build:
	docker build -t piston-engine-twin .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	rm -rf htmlcov/ .coverage dist/ build/ *.egg-info
