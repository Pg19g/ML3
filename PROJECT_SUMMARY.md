# ML3 Project Summary

## Overview

**ML3** is a production-grade, point-in-time (PIT) machine learning market data pipeline designed to prevent look-ahead bias in quantitative trading research. The system ingests market data from EODHD, applies rigorous temporal controls, engineers features, trains ML models, and provides backtesting capabilities—all through an intuitive dashboard and API.

## Key Features

### 1. Point-in-Time Data Integrity
- **Conservative Lags**: 60-day lag for quarterly reports, 90-day for annual reports
- **Trading Day Adjustment**: Rounds to next trading day + 2 extra days
- **Validity Intervals**: Each fundamental is valid from `as_of_date` until next report
- **Staleness Detection**: Flags data older than 540 days
- **Automatic Integrity Checks**: Validates no future data leakage

### 2. Data Ingestion
- **EODHD Integration**: Efficient API client with rate limiting and retry logic
- **Incremental Updates**: Idempotent merging with existing data
- **Bulk Operations**: Fetches multiple symbols in parallel
- **Data Quality**: Validates and cleans incoming data

### 3. Feature Engineering
- **Technical Features**: Returns, momentum, volatility, RSI, ATR, beta, skewness, kurtosis
- **Fundamental Features**: Profitability, leverage, liquidity, efficiency ratios
- **Proper Shifting**: All technical features shifted by 1 day to prevent leakage
- **Cross-Sectional Standardization**: Normalizes features per time period
- **Missing Value Handling**: Forward-fill with limits

### 4. Model Training
- **Algorithms**: LightGBM and XGBoost
- **Time-Series CV**: Purged K-Fold with 21-day embargo
- **Metrics**: IC, Rank IC, RMSE, MAE, R²
- **Model Registry**: Version control for trained models
- **Hyperparameter Tuning**: Optional Optuna integration

### 5. Backtesting
- **Rank-Based Strategy**: Long top decile, optional short bottom decile
- **Transaction Costs**: Configurable commission and slippage
- **Rebalancing**: Daily, weekly, or monthly
- **Performance Metrics**: Total return, CAGR, Sharpe, Sortino, max drawdown, win rate
- **Equity Curves**: Visualize strategy performance over time

### 6. Dashboard (Streamlit)
- **Data Management**: View stats, refresh data, build pipeline
- **Features & Labels**: Toggle features, inspect distributions
- **Training**: Configure and train models, view metrics
- **Backtest**: Run backtests, visualize results
- **Model Import**: Upload and validate external models
- **Diagnostics**: Check PIT integrity, staleness, coverage

### 7. API (FastAPI)
- **Model Management**: List, get info, delete models
- **Predictions**: Generate predictions for symbols/dates
- **Flow Triggers**: Programmatically trigger data pipelines
- **Data Stats**: Get data statistics and metadata

## Architecture

### Technology Stack

- **Language**: Python 3.11
- **Data Processing**: Pandas, Polars, DuckDB, PyArrow
- **ML Libraries**: LightGBM, XGBoost, scikit-learn, Optuna
- **Orchestration**: Prefect (flows for each pipeline step)
- **API**: FastAPI with Pydantic validation
- **Dashboard**: Streamlit with Plotly visualizations
- **Trading Calendar**: pandas-market-calendars
- **Model Export**: ONNX, joblib/pickle

### Project Structure

```
ML3/
├── config/              # YAML configuration files
│   ├── eodhd.yaml      # API settings
│   ├── universe.yaml   # Symbol universe
│   ├── pit.yaml        # Point-in-time rules
│   ├── features.yaml   # Feature definitions
│   ├── labels.yaml     # Label definitions
│   ├── train.yaml      # Training configuration
│   └── backtest.yaml   # Backtest settings
├── data/
│   ├── raw/            # Raw EODHD data (parquet)
│   ├── pit/            # Processed PIT data (parquet)
│   └── samples/        # Sample data files
├── models/
│   └── registry/       # Trained models + metadata
├── reports/            # Metrics, plots, backtest results
├── app/
│   ├── streamlit_app.py  # Dashboard (8501)
│   └── api.py          # FastAPI service (8000)
├── src/
│   ├── eodhd_client.py # EODHD API client
│   ├── calendars.py    # Trading calendar utilities
│   ├── pit.py          # Point-in-time logic
│   ├── features.py     # Feature engineering
│   ├── labels.py       # Label generation
│   ├── train.py        # Model training
│   ├── backtest.py     # Backtesting framework
│   ├── registry.py     # Model registry
│   ├── utils.py        # Utility functions
│   └── cli.py          # Command-line interface
├── flows/              # Prefect flows (orchestration)
│   ├── ingest_prices.py
│   ├── ingest_fundamentals.py
│   ├── build_pit.py
│   ├── build_features.py
│   ├── build_labels.py
│   ├── train.py
│   └── backtest.py
└── tests/              # Unit tests
    ├── test_pit.py
    ├── test_features.py
    └── test_leakage.py
```

## Data Flow

```
1. EODHD API
   ↓ (ingest_prices, ingest_fundamentals)
2. RAW Data
   - prices_daily.parquet (Date, Symbol, OHLCV)
   - fundamentals.parquet (Symbol, Period, Financials)
   ↓ (build_pit)
3. PIT Panel
   - daily_panel.parquet (with as_of_date, validity intervals)
   ↓ (build_features)
4. Features
   - features.parquet (technical + fundamental features)
   ↓ (build_labels)
5. Labels
   - labels.parquet (forward returns)
   ↓ (train)
6. Models
   - models/registry/{model_id}/ (model + metadata)
   ↓ (backtest)
7. Results
   - reports/{model_id}_* (metrics, equity curves)
```

## Usage Examples

### Quick Start

```bash
# 1. Install
git clone https://github.com/Pg19g/ML3.git
cd ML3
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with EODHD_API_KEY

# 3. Run pipeline
make ingest    # Ingest data
make build     # Build features/labels
make train     # Train model

# 4. Launch dashboard
make dashboard
```

### CLI Commands

```bash
# Data ingestion
python -m src.cli data ingest-prices
python -m src.cli data ingest-fundamentals

# Pipeline
python -m src.cli data build-pit
python -m src.cli features build
python -m src.cli labels build

# Training
python -m src.cli train run --model-type lightgbm

# Backtesting
python -m src.cli backtest run MODEL_ID

# Model management
python -m src.cli models list
python -m src.cli models info MODEL_ID

# Services
python -m src.cli dashboard  # Port 8501
python -m src.cli api        # Port 8000
```

### Python API

```python
from src.features import FeatureEngineer
from src.train import ModelTrainer
from src.registry import ModelRegistry

# Feature engineering
engineer = FeatureEngineer()
features_df = engineer.build_features(pit_panel)

# Model training
trainer = ModelTrainer()
cv_results = trainer.cross_validate(data, feature_cols, label_col)

# Model registry
registry = ModelRegistry()
models = registry.list_models()
predictions = registry.score_model(model_id, X)
```

## Configuration

All configuration is done via YAML files in `config/`:

- **eodhd.yaml**: API settings, rate limits, endpoints
- **universe.yaml**: Symbols to track, date ranges
- **pit.yaml**: PIT rules (lags, staleness, calendar)
- **features.yaml**: Feature definitions and parameters
- **labels.yaml**: Label definitions (horizons)
- **train.yaml**: Model types, CV scheme, hyperparameters
- **backtest.yaml**: Strategy, costs, rebalancing

## Testing

Comprehensive test suite covering:

- **Point-in-Time Logic**: As-of dates, validity intervals
- **Feature Engineering**: Technical indicators, fundamental ratios
- **Leakage Detection**: Ensures no future data in features
- **Data Integrity**: Validates PIT constraints

```bash
# Run all tests
pytest

# Run specific tests
pytest tests/test_pit.py
pytest tests/test_leakage.py -v

# With coverage
pytest --cov=src tests/
```

## Performance

- **Data Ingestion**: ~100 symbols in ~5 minutes (depends on API limits)
- **PIT Processing**: ~1M rows in ~30 seconds
- **Feature Engineering**: ~1M rows in ~2 minutes
- **Model Training**: 5-fold CV in ~5-10 minutes (depends on data size)
- **Backtesting**: ~1M predictions in ~1 minute

## Deployment

### Local Development
- Clone repository
- Install dependencies
- Configure `.env`
- Run pipeline

### Production
- Ubuntu 22.04 LTS
- Python 3.11
- Systemd services for dashboard and API
- Nginx reverse proxy
- SSL with Let's Encrypt
- Automated data refresh via cron

### Docker
- Dockerfile included
- Docker Compose for multi-service setup
- Persistent volumes for data/models

### Cloud
- AWS EC2, GCP Compute Engine, Azure VM
- 8GB+ RAM, 50GB+ storage
- Monitoring with CloudWatch/Stackdriver/Azure Monitor

## Documentation

- **README.md**: Project overview and quick start
- **userGuide.md**: Comprehensive user guide (660 lines)
- **DEPLOYMENT.md**: Deployment instructions (609 lines)
- **PROJECT_SUMMARY.md**: This file

## Code Statistics

- **Total Python Code**: ~4,600 lines
- **Source Modules**: 11 files
- **Flows**: 7 Prefect flows
- **Tests**: 3 test files
- **Configuration**: 7 YAML files
- **Documentation**: 4 markdown files

## Key Innovations

1. **Rigorous PIT Logic**: Conservative lags + trading day adjustments
2. **Automatic Leakage Prevention**: Shifted features, as-of joins
3. **Purged K-Fold CV**: Embargo periods prevent temporal leakage
4. **Model Registry**: Version control and schema validation
5. **External Model Import**: Deploy ONNX/joblib models
6. **Interactive Dashboard**: No-code interface for entire pipeline
7. **Configuration-Driven**: Zero code changes for common adjustments

## Use Cases

1. **Quantitative Research**: Develop and test trading strategies
2. **Factor Analysis**: Identify predictive features
3. **Model Development**: Train and validate ML models
4. **Backtesting**: Evaluate strategy performance
5. **Production Trading**: Deploy models via API
6. **Education**: Learn point-in-time data handling

## Limitations

1. **Data Source**: Currently only supports EODHD (extensible)
2. **Asset Class**: Equities only (can be extended)
3. **Strategy Types**: Rank-based strategies (can be extended)
4. **Execution**: Assumes perfect execution (no market impact)
5. **Costs**: Simplified transaction cost model

## Future Enhancements

1. **Additional Data Sources**: Alpha Vantage, Quandl, etc.
2. **More Asset Classes**: Futures, options, crypto
3. **Advanced Strategies**: Market-making, pairs trading
4. **Real-Time Data**: Streaming data support
5. **Portfolio Optimization**: Risk parity, mean-variance
6. **Execution Simulation**: Realistic market impact
7. **Multi-Asset Backtesting**: Cross-asset strategies

## License

MIT License

## Repository

GitHub: https://github.com/Pg19g/ML3

## Support

- **Issues**: GitHub Issues
- **Documentation**: See README.md, userGuide.md, DEPLOYMENT.md
- **Email**: support@example.com

---

**Disclaimer**: This is a research and educational tool. Always validate results and consult with financial professionals before making investment decisions. Past performance does not guarantee future results.
