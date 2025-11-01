# ML3 - Point-in-Time ML Market Data Pipeline

A production-grade, point-in-time (PIT) market data pipeline and ML experimentation stack that ingests EODHD data, prevents look-ahead bias, and provides a dashboard for model management and deployment.

## Features

- **Point-in-Time Data Processing**: Rigorous as-of logic with conservative lags to prevent look-ahead bias
- **EODHD Integration**: Efficient ingestion of EOD prices and fundamentals with rate limiting and retry logic
- **Feature Engineering**: Technical indicators and fundamental ratios with proper shifting
- **Time-Series Cross-Validation**: Purged K-Fold with embargo periods
- **Backtesting Framework**: Rank-based strategies with transaction costs and slippage
- **Model Registry**: Import, validate, and deploy external models (ONNX, joblib)
- **Interactive Dashboard**: Streamlit UI for data management, training, and diagnostics
- **API Service**: FastAPI endpoints for programmatic access

## Quick Start

### Prerequisites

- Python 3.11+
- EODHD API key ([get one here](https://eodhd.com/))

### Installation

**For Mac M3 Pro users**: See [INSTALL_MAC_M3.md](INSTALL_MAC_M3.md) for Mac-specific instructions.

```bash
# Clone the repository
git clone https://github.com/Pg19g/ML3.git
cd ML3

# Install dependencies using pip
pip install -r requirements.txt

# For Mac M3 Pro (Apple Silicon)
pip install -r requirements-mac-m3.txt

# Or using poetry
poetry install

# Set up environment variables
cp .env.example .env
# Edit .env and add your EODHD_API_KEY

# Configure Prefect (IMPORTANT!)
./setup_prefect.sh

# Load environment variables
source .env
export $(cat .env | xargs)
```

### Prefect Setup

**IMPORTANT**: Before running any pipeline commands, configure Prefect to run in ephemeral mode (no server required).

**Quick Setup**:
```bash
./setup_prefect.sh
```

**Manual Setup**:
Add these to your `.env` file:
```
PREFECT_API_DATABASE_CONNECTION_URL=sqlite+aiosqlite:///$HOME/.prefect/prefect.db
PREFECT_API_URL=
PREFECT_LOGGING_LEVEL=INFO
```

Then load the environment:
```bash
source .env
export $(cat .env | xargs)
```

For detailed instructions, see [PREFECT_SETUP.md](PREFECT_SETUP.md).

**Why?** By default, Prefect tries to connect to a server. Local mode with SQLite runs flows locally without a server, which is perfect for development.

### Basic Usage

```bash
# 1. Ingest price data
python -m src.cli data ingest-prices

# 2. Ingest fundamental data
python -m src.cli data ingest-fundamentals

# 3. Build point-in-time panel
python -m src.cli data build-pit

# 4. Build features
python -m src.cli features build

# 5. Build labels
python -m src.cli labels build

# 6. Train a model
python -m src.cli train run --model-type lightgbm

# 7. Run backtest (replace MODEL_ID with actual model ID)
python -m src.cli backtest run MODEL_ID

# 8. Launch dashboard
python -m src.cli dashboard

# 9. Launch API service
python -m src.cli api
```

## Architecture

### Data Flow

```
EODHD API
    ↓
RAW Data (prices_daily.parquet, fundamentals.parquet)
    ↓
PIT Processing (as-of logic, validity intervals)
    ↓
PIT Panel (daily_panel.parquet)
    ↓
Feature Engineering (technical + fundamental features)
    ↓
Features (features.parquet)
    ↓
Label Generation (forward returns)
    ↓
Labels (labels.parquet)
    ↓
Model Training (LightGBM/XGBoost with Purged K-Fold)
    ↓
Model Registry
    ↓
Backtesting (rank-based strategies)
    ↓
Reports & Metrics
```

### Directory Structure

```
.
├── config/              # YAML configuration files
│   ├── eodhd.yaml      # API settings
│   ├── universe.yaml   # Symbol universe
│   ├── pit.yaml        # Point-in-time rules
│   ├── features.yaml   # Feature definitions
│   ├── labels.yaml     # Label definitions
│   ├── train.yaml      # Training configuration
│   └── backtest.yaml   # Backtest settings
├── data/
│   ├── raw/            # Raw EODHD data
│   ├── pit/            # Processed PIT data
│   └── samples/        # Sample data files
├── models/
│   └── registry/       # Trained models
├── reports/            # Metrics and results
├── app/
│   ├── streamlit_app.py  # Dashboard
│   └── api.py          # FastAPI service
├── src/
│   ├── eodhd_client.py # EODHD API client
│   ├── calendars.py    # Trading calendar utilities
│   ├── pit.py          # Point-in-time logic
│   ├── features.py     # Feature engineering
│   ├── labels.py       # Label generation
│   ├── train.py        # Model training
│   ├── backtest.py     # Backtesting
│   ├── registry.py     # Model registry
│   ├── utils.py        # Utility functions
│   └── cli.py          # Command-line interface
├── flows/              # Prefect flows
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

## Point-in-Time (PIT) Rules

### Fundamental Availability

Fundamentals become available based on conservative lags:

1. **As-of Date Calculation**:
   - If `filing_date` exists: `as_of_date = max(filing_date, period_end + lag_days)`
   - Otherwise: `as_of_date = period_end + lag_days`
   - Quarterly reports: `lag_days = 60`
   - Annual reports: `lag_days = 90`

2. **Trading Day Adjustment**:
   - Round `as_of_date` up to next trading day
   - Add `extra_trading_lag = 2` trading days

3. **Validity Intervals**:
   - `valid_from = as_of_date`
   - `valid_to = next_as_of_date - 1 day`

4. **Staleness**:
   - Mark fundamentals as stale after `stale_max_days = 540` days
   - Add `is_stale_fund` flag

### Technical Features

All technical features are computed locally and shifted by 1 trading day to prevent leakage:

- Daily returns, momentum (21/63/126 days)
- Volatility (20/63 days rolling std)
- ATR(14), RSI(14)
- Rolling beta to market (252 days)
- Skewness and kurtosis

### Labels

Forward returns are computed by negative shifting:
- `ret_1d_fwd = AdjClose[t+1] / AdjClose[t] - 1`
- `ret_5d_fwd`, `ret_21d_fwd`, etc.

## Configuration

All configuration is done via YAML files in the `config/` directory. Key settings:

### Universe (`universe.yaml`)

Define the symbols to track:

```yaml
symbols:
  - AAPL.US
  - MSFT.US
  - GOOGL.US
  # ... more symbols

date_range:
  start: "2020-01-01"
  end: null  # null = today
```

### PIT Rules (`pit.yaml`)

Control fundamental availability:

```yaml
q_lag_days: 60
y_lag_days: 90
extra_trading_lag: 2
stale_max_days: 540
calendar: "NYSE"
```

### Features (`features.yaml`)

Enable/disable features and set parameters:

```yaml
technical:
  - name: ret_1d
    type: return
    window: 1
    enabled: true
  # ... more features

fundamental:
  - name: net_margin
    type: ratio
    numerator: "NetIncome"
    denominator: "TotalRevenue"
    enabled: true
  # ... more features
```

### Training (`train.yaml`)

Configure model training:

```yaml
models:
  lightgbm:
    enabled: true
    params:
      objective: "regression"
      num_leaves: 31
      learning_rate: 0.05
      # ... more params

cv:
  method: "purged_kfold"
  n_splits: 5
  embargo_days: 21

random_seed: 42
```

### Backtesting (`backtest.yaml`)

Set up trading strategy:

```yaml
strategy:
  type: "rank_based"
  long_top_pct: 0.1  # Long top 10%
  short_bottom_pct: 0.0  # Long-only
  weighting: "equal"
  rebalance_frequency: "weekly"

costs:
  commission_pct: 0.001  # 0.1%
  slippage_pct: 0.0005   # 0.05%
```

## CLI Reference

### Data Commands

```bash
# Ingest prices
python -m src.cli data ingest-prices [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--incremental/--full]

# Ingest fundamentals
python -m src.cli data ingest-fundamentals [--incremental/--full]

# Build PIT panel
python -m src.cli data build-pit
```

### Feature Commands

```bash
# Build features
python -m src.cli features build
```

### Label Commands

```bash
# Build labels
python -m src.cli labels build
```

### Training Commands

```bash
# Train model
python -m src.cli train run [--model-type lightgbm|xgboost]
```

### Backtest Commands

```bash
# Run backtest
python -m src.cli backtest run MODEL_ID
```

### Model Registry Commands

```bash
# List models
python -m src.cli models list

# Show model info
python -m src.cli models info MODEL_ID

# Delete model
python -m src.cli models delete MODEL_ID
```

### Dashboard & API

```bash
# Launch Streamlit dashboard
python -m src.cli dashboard [--port 8501]

# Launch FastAPI service
python -m src.cli api [--host 0.0.0.0] [--port 8000]
```

## Dashboard

The Streamlit dashboard provides a web interface for:

1. **Data Management**: View statistics, refresh data, build PIT panel
2. **Features & Labels**: Toggle features, inspect distributions
3. **Training**: Configure and train models, view metrics
4. **Backtest**: Run backtests, visualize equity curves
5. **Model Import**: Upload and validate external models
6. **Diagnostics**: Check PIT integrity, data staleness, coverage

Access at: `http://localhost:8501`

## API Endpoints

The FastAPI service exposes:

- `GET /models` - List all models
- `GET /models/{model_id}` - Get model info
- `POST /models/{model_id}/predict` - Generate predictions
- `DELETE /models/{model_id}` - Delete model
- `POST /flows/trigger` - Trigger Prefect flows
- `GET /data/stats` - Get data statistics

API docs at: `http://localhost:8000/docs`

## Testing

Run tests with pytest:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_pit.py

# Run with coverage
pytest --cov=src tests/

# Run leakage tests
pytest tests/test_leakage.py -v
```

## Model Import

Import external models (ONNX or joblib format):

```bash
# Via dashboard: Upload file in "Model Import" page

# Via API:
curl -X POST "http://localhost:8000/models/import" \
  -F "file=@model.onnx" \
  -F "model_id=my_model" \
  -F "feature_cols=ret_1d,vol_20d,..."
```

Models are validated against the feature schema before deployment.

## Reproducibility

All runs are deterministic and reproducible:

- Fixed random seeds in `train.yaml`
- Pinned dependencies in `pyproject.toml`
- Idempotent, incremental ETL
- Configuration-driven (no code changes needed)

## Best Practices

1. **Always check PIT integrity** after building the panel
2. **Run leakage tests** before training models
3. **Use incremental ingestion** to avoid API rate limits
4. **Monitor data staleness** in the diagnostics page
5. **Validate external models** before deployment
6. **Document configuration changes** in version control

## Troubleshooting

### No data ingested

- Check EODHD API key in `.env`
- Verify symbols in `universe.yaml` are valid
- Check API rate limits

### PIT integrity check fails

- Review `pit.yaml` configuration
- Check for missing `filing_date` in fundamentals
- Verify trading calendar is correct

### Training fails

- Ensure features and labels are built
- Check for sufficient data (min 100 observations)
- Review feature/label configuration

### High data staleness

- Increase `stale_max_days` in `pit.yaml`
- Refresh fundamental data more frequently
- Check for delisted symbols

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run `pytest` and ensure all tests pass
5. Run `black` and `ruff` for code formatting
6. Submit a pull request

## License

MIT License

## Acknowledgments

- EODHD for market data API
- Prefect for workflow orchestration
- LightGBM and XGBoost for ML models
- Streamlit for dashboard framework

## Support

For issues and questions:
- GitHub Issues: [repository-url]/issues
- Documentation: [repository-url]/wiki
- Email: support@example.com

---

**Note**: This is a research and educational tool. Always validate results and consult with financial professionals before making investment decisions.
