.PHONY: help install test lint format check clean dashboard api

help:
	@echo "ML3 - Point-in-Time ML Market Data Pipeline"
	@echo ""
	@echo "Available commands:"
	@echo "  make install       Install dependencies"
	@echo "  make test          Run tests"
	@echo "  make lint          Run linting"
	@echo "  make format        Format code"
	@echo "  make check         Run linting and tests"
	@echo "  make clean         Clean generated files"
	@echo "  make dashboard     Launch Streamlit dashboard"
	@echo "  make api           Launch FastAPI service"
	@echo ""
	@echo "Data pipeline:"
	@echo "  make ingest        Ingest prices and fundamentals"
	@echo "  make build         Build PIT panel, features, and labels"
	@echo "  make train         Train a model"
	@echo ""

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

lint:
	ruff check src/ flows/ app/ tests/
	black --check src/ flows/ app/ tests/

format:
	black src/ flows/ app/ tests/
	ruff check --fix src/ flows/ app/ tests/

check: lint test

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

dashboard:
	python -m src.cli dashboard

api:
	python -m src.cli api

# Data pipeline shortcuts
ingest:
	python -m src.cli data ingest-prices
	python -m src.cli data ingest-fundamentals

build:
	python -m src.cli data build-pit
	python -m src.cli features build
	python -m src.cli labels build

train:
	python -m src.cli train run --model-type lightgbm
