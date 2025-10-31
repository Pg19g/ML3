# ML3 User Guide

Welcome to ML3, a production-grade point-in-time ML market data pipeline! This guide will help you get started and make the most of the platform.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Configuration](#configuration)
3. [Data Pipeline](#data-pipeline)
4. [Model Training](#model-training)
5. [Backtesting](#backtesting)
6. [Dashboard Usage](#dashboard-usage)
7. [API Usage](#api-usage)
8. [Advanced Topics](#advanced-topics)
9. [Troubleshooting](#troubleshooting)

## Getting Started

### Step 1: Installation

```bash
# Clone the repository
git clone https://github.com/Pg19g/ML3.git
cd ML3

# Install dependencies
pip install -r requirements.txt

# Or use make
make install
```

### Step 2: Configuration

Create your environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your EODHD API key:

```
EODHD_API_KEY=your_api_key_here
TZ=UTC
```

Get your API key from [EODHD](https://eodhd.com/).

### Step 3: Configure Universe

Edit `config/universe.yaml` to define which stocks to track:

```yaml
symbols:
  - AAPL.US
  - MSFT.US
  - GOOGL.US
  # Add more symbols...

date_range:
  start: "2020-01-01"
  end: null  # null = today
```

### Step 4: Run Your First Pipeline

```bash
# Quick start - run the full pipeline
make ingest    # Ingest data
make build     # Build features and labels
make train     # Train a model
```

Or step by step:

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
```

### Step 5: Launch Dashboard

```bash
python -m src.cli dashboard
```

Open your browser to `http://localhost:8501`

## Configuration

### Universe Configuration (`config/universe.yaml`)

Define which stocks to track and the date range:

```yaml
symbols:
  - AAPL.US
  - MSFT.US
  # ... more symbols

date_range:
  start: "2020-01-01"
  end: null  # null = today

rules:
  min_market_cap: 1000000000  # $1B
  min_avg_volume: 100000
  exchanges:
    - US
    - NASDAQ

include_delisted: true  # Reduce survivorship bias
```

### PIT Rules (`config/pit.yaml`)

Control how fundamentals become available:

```yaml
q_lag_days: 60        # Quarterly report lag
y_lag_days: 90        # Annual report lag
extra_trading_lag: 2  # Additional trading days
stale_max_days: 540   # Mark as stale after 18 months
calendar: "NYSE"      # Trading calendar
```

**What this means:**
- Quarterly reports become available 60 days after period end (if no filing date)
- Annual reports become available 90 days after period end
- Add 2 more trading days for processing
- Mark data as stale after 540 days

### Features (`config/features.yaml`)

Enable/disable features and set parameters:

```yaml
technical:
  - name: ret_1d
    type: return
    window: 1
    enabled: true
    
  - name: mom_21d
    type: momentum
    window: 21
    enabled: true
    
  - name: vol_20d
    type: volatility
    window: 20
    enabled: true

fundamental:
  - name: net_margin
    type: ratio
    numerator: "NetIncome"
    denominator: "TotalRevenue"
    enabled: true
```

**Feature Types:**
- `return`: Price returns over window
- `momentum`: Price momentum over window
- `volatility`: Rolling standard deviation
- `rsi`: Relative Strength Index
- `atr`: Average True Range
- `beta`: Rolling beta to market
- `ratio`: Fundamental ratios

### Labels (`config/labels.yaml`)

Define prediction targets:

```yaml
labels:
  - name: ret_1d_fwd
    type: forward_return
    horizon: 1
    enabled: true
    
  - name: ret_5d_fwd
    type: forward_return
    horizon: 5
    enabled: true
```

### Training (`config/train.yaml`)

Configure model training:

```yaml
models:
  lightgbm:
    enabled: true
    params:
      objective: "regression"
      num_leaves: 31
      learning_rate: 0.05

cv:
  method: "purged_kfold"
  n_splits: 5
  embargo_days: 21

random_seed: 42
```

### Backtesting (`config/backtest.yaml`)

Set up trading strategy:

```yaml
strategy:
  type: "rank_based"
  long_top_pct: 0.1      # Long top 10%
  short_bottom_pct: 0.0  # Long-only
  weighting: "equal"
  rebalance_frequency: "weekly"

costs:
  commission_pct: 0.001  # 0.1% per trade
  slippage_pct: 0.0005   # 0.05% slippage
```

## Data Pipeline

### Understanding the Pipeline

```
EODHD API → RAW Data → PIT Processing → Features → Labels → Models
```

### Step 1: Data Ingestion

**Ingest Prices:**

```bash
# Full refresh
python -m src.cli data ingest-prices --full

# Incremental (default)
python -m src.cli data ingest-prices --incremental

# Specific date range
python -m src.cli data ingest-prices --start-date 2023-01-01 --end-date 2023-12-31
```

**Ingest Fundamentals:**

```bash
python -m src.cli data ingest-fundamentals
```

**What happens:**
- Downloads EOD prices (OHLCV) for all symbols
- Downloads fundamentals (balance sheet, income statement, cash flow)
- Stores in `data/raw/prices_daily.parquet` and `data/raw/fundamentals.parquet`
- Incremental mode merges with existing data (idempotent)

### Step 2: PIT Processing

```bash
python -m src.cli data build-pit
```

**What happens:**
- Computes `as_of_date` for each fundamental row
- Creates validity intervals (`valid_from`, `valid_to`)
- Performs as-of join with daily price data
- Adds staleness flags
- Stores in `data/pit/daily_panel.parquet`

**PIT Integrity:**
The system ensures no future data leakage by:
1. Conservative lags (60/90 days + 2 trading days)
2. Validity intervals prevent using future fundamentals
3. Automatic integrity checks

### Step 3: Feature Engineering

```bash
python -m src.cli features build
```

**What happens:**
- Computes technical features from prices
- Computes fundamental ratios
- Shifts all technical features by 1 day (prevent leakage)
- Applies cross-sectional standardization
- Handles missing values
- Stores in `data/pit/features.parquet`

### Step 4: Label Generation

```bash
python -m src.cli labels build
```

**What happens:**
- Computes forward returns (1d, 5d, 21d, etc.)
- Uses negative shift (future values)
- Validates label quality
- Stores in `data/pit/labels.parquet`

## Model Training

### Training a Model

```bash
# Train LightGBM (default)
python -m src.cli train run

# Train XGBoost
python -m src.cli train run --model-type xgboost
```

**What happens:**
1. Loads features and labels
2. Splits data using Purged K-Fold (5 folds, 21-day embargo)
3. Trains model on each fold
4. Computes metrics (IC, Rank IC, RMSE, R²)
5. Saves model to `models/registry/`
6. Saves metrics to `reports/`

### Understanding Metrics

**Information Coefficient (IC):**
- Pearson correlation between predictions and actual returns
- Range: -1 to 1
- Good: > 0.05
- Excellent: > 0.10

**Rank IC:**
- Spearman correlation (rank-based)
- More robust to outliers
- Good: > 0.05

**RMSE/MAE:**
- Regression error metrics
- Lower is better

### Cross-Validation

The system uses **Purged K-Fold** with embargo:

```
Fold 1: Train [----] Test [--] Embargo [==]
Fold 2:        Train [----] Test [--] Embargo [==]
Fold 3:               Train [----] Test [--] Embargo [==]
```

- **Purging**: No overlap between train and test
- **Embargo**: 21-day gap to prevent leakage
- **Time-series**: Respects temporal order

## Backtesting

### Running a Backtest

```bash
# List models
python -m src.cli models list

# Run backtest
python -m src.cli backtest run MODEL_ID
```

**What happens:**
1. Loads model and generates predictions
2. Ranks stocks cross-sectionally each day
3. Longs top 10% (configurable)
4. Rebalances weekly (configurable)
5. Applies transaction costs and slippage
6. Computes performance metrics
7. Saves results to `reports/`

### Understanding Backtest Metrics

**Total Return:**
- Cumulative return over backtest period

**CAGR (Compound Annual Growth Rate):**
- Annualized return
- Good: > 10%

**Sharpe Ratio:**
- Risk-adjusted return (return / volatility)
- Good: > 1.0
- Excellent: > 2.0

**Max Drawdown:**
- Largest peak-to-trough decline
- Lower is better
- Good: < 20%

**Win Rate:**
- Percentage of profitable periods
- Good: > 50%

**Turnover:**
- Average portfolio turnover
- Lower is better (less trading costs)

## Dashboard Usage

Launch the dashboard:

```bash
python -m src.cli dashboard
```

### Data Page

- **View Statistics**: See row counts, date ranges
- **Refresh Data**: Trigger data ingestion
- **Build Pipeline**: Build PIT panel, features, labels
- **PIT Integrity**: Check for data leakage

### Features & Labels Page

- **Toggle Features**: Enable/disable features
- **View Distributions**: Inspect feature distributions
- **Summary Statistics**: Mean, std, min, max, etc.

### Training Page

- **Configure Training**: Select model type
- **Start Training**: Train new models
- **View Results**: See metrics, feature importance
- **Recent Models**: Browse trained models

### Backtest Page

- **Select Model**: Choose model to backtest
- **Configure Strategy**: Set long/short percentages
- **Run Backtest**: Execute backtest
- **View Results**: Equity curve, metrics, drawdowns

### Model Import Page

- **Upload Model**: Import ONNX or joblib models
- **Validate Schema**: Check feature compatibility
- **Deploy Model**: Add to registry

### Diagnostics Page

- **Data Staleness**: Check fundamental data age
- **Coverage**: Missing data by feature
- **PIT Integrity**: Leakage detection

## API Usage

Launch the API:

```bash
python -m src.cli api
```

API documentation: `http://localhost:8000/docs`

### List Models

```bash
curl http://localhost:8000/models
```

### Get Model Info

```bash
curl http://localhost:8000/models/MODEL_ID
```

### Generate Predictions

```bash
curl -X POST http://localhost:8000/models/MODEL_ID/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "MODEL_ID",
    "symbols": ["AAPL.US", "MSFT.US"],
    "date": "2023-12-01"
  }'
```

### Trigger Flows

```bash
curl -X POST http://localhost:8000/flows/trigger \
  -H "Content-Type: application/json" \
  -d '{
    "flow_name": "ingest-prices",
    "parameters": {"incremental": true}
  }'
```

## Advanced Topics

### Custom Features

Add custom features in `src/features.py`:

```python
def compute_custom_feature(self, df: pd.DataFrame) -> pd.DataFrame:
    """Compute custom feature."""
    result = df.copy()
    result['my_feature'] = ...  # Your logic here
    return result
```

Update `config/features.yaml`:

```yaml
technical:
  - name: my_feature
    type: custom
    enabled: true
```

### External Model Import

Import models trained outside the pipeline:

```python
from src.registry import ModelRegistry

registry = ModelRegistry()
registry.import_external_model(
    model_path='path/to/model.onnx',
    model_id='my_external_model',
    feature_cols=['ret_1d', 'vol_20d', ...],
    label_col='ret_5d_fwd'
)
```

### Hyperparameter Tuning

Enable Optuna in `config/train.yaml`:

```yaml
optuna:
  enabled: true
  n_trials: 50
  timeout: 3600
  
  search_space:
    lightgbm:
      num_leaves: [20, 100]
      learning_rate: [0.01, 0.1]
```

### Custom Backtesting Strategies

Modify `src/backtest.py` to implement custom strategies:

```python
def custom_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
    """Custom trading strategy."""
    # Your strategy logic here
    return df_with_positions
```

## Troubleshooting

### Problem: No data ingested

**Solution:**
- Check EODHD API key in `.env`
- Verify symbols in `universe.yaml` are valid
- Check API rate limits
- Try with a smaller universe first

### Problem: PIT integrity check fails

**Solution:**
- Review `pit.yaml` configuration
- Check for missing `filing_date` in fundamentals
- Verify trading calendar is correct
- Increase lag days if needed

### Problem: Training fails

**Solution:**
- Ensure features and labels are built
- Check for sufficient data (min 100 observations)
- Review feature/label configuration
- Check for NaN values in features

### Problem: High data staleness

**Solution:**
- Increase `stale_max_days` in `pit.yaml`
- Refresh fundamental data more frequently
- Check for delisted symbols
- Filter out stale data in training

### Problem: Poor model performance

**Solution:**
- Check IC and Rank IC metrics
- Ensure no data leakage (run tests)
- Try different feature combinations
- Adjust CV parameters
- Increase training data

### Problem: API rate limits

**Solution:**
- Use incremental ingestion
- Reduce universe size
- Adjust rate limits in `eodhd.yaml`
- Spread ingestion over time

## Best Practices

1. **Start Small**: Test with 5-10 symbols first
2. **Check Integrity**: Always verify PIT integrity
3. **Run Tests**: Execute leakage tests regularly
4. **Monitor Staleness**: Keep fundamentals fresh
5. **Version Control**: Track configuration changes
6. **Document Changes**: Note why you changed parameters
7. **Validate Models**: Check metrics before deployment
8. **Backtest Thoroughly**: Test on out-of-sample data
9. **Monitor Performance**: Track live vs backtest results
10. **Stay Updated**: Refresh data regularly

## Getting Help

- **Documentation**: See `README.md`
- **Issues**: GitHub Issues
- **Tests**: Run `pytest tests/` for examples
- **Code**: Review source code in `src/`

## Next Steps

1. **Customize Universe**: Add your target stocks
2. **Tune Features**: Enable/disable features
3. **Experiment**: Try different models and parameters
4. **Backtest**: Validate performance
5. **Deploy**: Use API for production predictions

---

Happy trading! Remember: This is a research tool. Always validate results and consult with financial professionals before making investment decisions.
