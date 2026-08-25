.PHONY: setup lint test clean all

# Developer Experience Makefile
# Standardizes environment setup and testing for the MLOps Engine

setup:
	@echo "🚀 Setting up the environment using uv..."
	uv venv
	uv pip install -e .[dev]

lint:
	@echo "🧹 Running ruff linter..."
	uv run ruff check src tests --ignore EXE002,BLE001,RUF015,DTZ005
	uv run ruff format --check src tests

test:
	@echo "🧪 Running unit tests..."
	uv run pytest tests/ -v

clean:
	@echo "🗑️ Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

all: setup lint test
