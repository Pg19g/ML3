.PHONY: help install install-dev test test-quick test-coverage lint format check clean dashboard api backtest diagnostics all

help:
	@echo "ML3 - Point-in-Time ML Market Data Pipeline"
	@echo ""
	@echo "Available commands:"
	@echo "  make install       Install dependencies"
	@echo "  make install-dev   Install dev dependencies"
	@echo "  make test          Run all tests"
	@echo "  make test-quick    Run tests (stop on first failure)"
	@echo "  make test-coverage Run tests with coverage report"
	@echo "  make lint          Run linting"
	@echo "  make format        Format code"
	@echo "  make check         Run linting and tests"
	@echo "  make clean         Clean generated files"
	@echo "  make dashboard     Launch Streamlit dashboard"
	@echo "  make api           Launch FastAPI service"
	@echo "  make backtest      Run backtest"
	@echo "  make diagnostics   Run diagnostics"
	@echo ""
	@echo "Data pipeline:"
	@echo "  make ingest        Ingest prices and fundamentals"
	@echo "  make build         Build PIT panel, features, and labels"
	@echo "  make train         Train a model"
	@echo "  make all           Run complete pipeline (ingest -> build -> train -> backtest)"
	@echo ""

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov ruff black

test:
	pytest tests/ -v --tb=short

test-quick:
	pytest tests/ -v --tb=short -x

test-coverage:
	pytest tests/ -v --cov=src --cov=flows --cov-report=html --cov-report=term

lint:
	@echo "Running ruff..."
	@ruff check src/ flows/ app/ tests/ || true
	@echo "Running black..."
	@black --check src/ flows/ app/ tests/ || true

format:
	black src/ flows/ app/ tests/
	ruff check --fix src/ flows/ app/ tests/ || true

check: lint test

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage 2>/dev/null || true

dashboard:
	streamlit run app/streamlit_app.py

api:
	uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

# Data pipeline shortcuts
ingest:
	@echo "Ingesting prices..."
	python -m src.cli data ingest-prices
	@echo "Ingesting fundamentals..."
	python -m src.cli data ingest-fundamentals

build:
	@echo "Building PIT panel..."
	python -m src.cli data build-pit
	@echo "Building features..."
	python -m src.cli features build
	@echo "Building labels..."
	python -m src.cli labels build

train:
	@echo "Training model..."
	python -m src.cli train run --model-type lightgbm

backtest:
	@echo "Running backtest..."
	python -m src.cli backtest run

diagnostics:
	@echo "Running diagnostics..."
	streamlit run app/streamlit_diagnostics.py

# Complete pipeline
all: ingest build train backtest
	@echo "✅ Complete pipeline finished!"
	@echo "Run 'make dashboard' to view results"
