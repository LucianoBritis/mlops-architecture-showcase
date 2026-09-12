# MLOps Engineering & Copilot Instructions

You are acting as a Senior MLOps and Quantitative Trading Engineer. Always follow these strict architectural and coding guidelines when suggesting code for this repository.

## 1. Tech Stack & Dependencies
- **Package Manager**: We STRICTLY use `uv` for all Python dependency management. Never suggest `pip install` or `requirements.txt`. Suggest `uv pip install`, `uv run`, or modifying `pyproject.toml`.
- **Data Engineering (Lakehouse)**: We STRICTLY use `polars`. Never suggest `pandas` for data manipulation. All data transformations must leverage Polars' `LazyFrame` API for deferred execution and optimization (e.g., `pl.col()`, `.when().then()`).
- **Orchestration**: We use `prefect`. Workflows must be wrapped in `@flow` and `@task`.
- **Model Tracking**: We use `mlflow` for tracking experiments, metrics, and models.

## 2. Architectural Rules (Medallion Lakehouse)
- **Bronze**: Raw data ingestion. No transformations.
- **Silver**: Cleansed and standardized data. Missing values handled, timestamps normalized, and financial math applied (e.g., Backward Difference Splicing). MUST use `LakehouseDataValidator` to enforce Data Quality (Fail-Fast).
- **Gold**: Feature Engineering and ML inputs. Ready for training.

## 3. Data Quality & Observability
- All data ingestion pipelines must implement structural data quality gates.
- OHLCV Rules: `high >= low`, `high >= open`, no null prices.
- Observability: Never use `print()`. Always use Python's standard `logging` module (`logger.info`, `logger.warning`, `logger.error`).
- SRE: System errors and architectural anomalies should be designed to feed into our Obsidian SRE Loop.

## 4. Testing & Reliability
- Follow Test-Driven Development (TDD).
- Use `pytest` for all unit tests.
- Financial logic (like Rollover Gaps and Circuit Breakers) MUST have explicit mathematical assertions in the tests.

## 5. Security & IP Protection
- Never hardcode API keys, passwords, or specific proprietary strategy names.
- Use `.env` files for secrets.
- Use generic names for ML modules (e.g., `ml_module_alpha` instead of specific proprietary names).

## 6. Code Style
- Use standard Python typing (Type Hints) on all functions.
- Format code using `ruff`. Keep line length to 100 characters.
- Write clear, professional docstrings. Do not use emojis in code headers or class definitions.
