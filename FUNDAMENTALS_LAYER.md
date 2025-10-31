## Production-Grade Fundamentals Layer

This document describes the production-grade fundamentals layer implemented in ML3, covering versioning, restatements, quarterly vs annual precedence, strict PIT interval joins, and comprehensive testing.

---

### Overview

The fundamentals layer ensures rigorous point-in-time (PIT) semantics for financial statement data, preventing look-ahead bias and handling real-world complexities like restatements, quarterly vs annual reporting, and data staleness.

**Key Features**:
- **Versioning**: Tracks restatements and corrections via `updated_at` timestamps
- **SCD-2 Intervals**: Slowly Changing Dimension Type 2 validity windows
- **Precedence Policies**: Configurable quarterly vs annual handling
- **Staleness Detection**: Nulls out fundamentals older than threshold
- **Leakage Prevention**: Validates that no future data is used
- **Ratio Computation**: Computes metrics at filing level, not daily
- **TTM Support**: Trailing Twelve Months ratios for quarterly data

---

### Data Model

#### Raw Fundamentals (`data/raw/fundamentals.parquet`)

**Schema**:
- `symbol` (str): Ticker symbol
- `statement_type` (str): 'quarterly' or 'annual'
- `period_end` (date): Period end date
- `filing_date` (date): Filing date (optional)
- `updated_at` (datetime): Restatement/correction timestamp (optional)
- `report_currency` (str): Currency code (optional)
- `audited` (bool): Audit status (optional)
- Raw numeric fields: `TotalRevenue`, `NetIncome`, `TotalAssets`, etc.

**Primary Key**: `(symbol, statement_type, period_end, filing_date)`

---

#### Fundamentals Intervals (`data/pit/fundamentals_intervals.parquet`)

**Schema**:
- `symbol` (str): Ticker symbol
- `statement_type` (str): 'quarterly' or 'annual'
- `period_end` (date): Period end date
- `version_id` (int): Sequential version number per report
- `as_of_date` (date): When original report became available
- `effective_from` (date): When this version became available
- `valid_from` (date): Start of validity interval
- `valid_to` (date): End of validity interval
- Derived ratios: `gross_margin`, `roe`, `debt_to_equity`, etc.
- TTM ratios: `gross_margin_ttm`, `roe_ttm`, etc. (quarterly only)
- `report_currency` (str): Currency code
- `audited` (bool): Audit status

**Purpose**: Defines when each version of each fundamental report is valid.

---

#### Daily Panel (`data/pit/daily_panel.parquet`)

**Schema**:
- `symbol` (str): Ticker symbol
- `date` (date): Trading date
- `source_ts_price` (datetime): When price data became available
- `source_ts_fund` (datetime): When fundamental data became available
- `is_stale_fund` (bool): Whether fundamental is stale
- `days_since_fund` (int): Days since fundamental became valid
- Price fields: `open`, `high`, `low`, `close`, `adj_close`, `volume`
- Fundamental fields: Latest valid fundamental data as of `date`

**Purpose**: Complete daily panel with prices and fundamentals joined via PIT logic.

---

### Configuration

Configuration is in `config/pit_enhanced.yaml`:

```yaml
# Lag days for computing as_of_date
q_lag_days: 60          # Quarterly: 60 days after period_end
y_lag_days: 90          # Annual: 90 days after period_end

# Extra trading days after rounding
extra_trading_lag: 2

# Staleness threshold
stale_max_days: 540     # ~18 months

# Precedence policy
precedence: quarter_over_annual  # or 'both_suffixes'

# Versioning
enable_versioning: true

# Interval join method
join_method: polars     # or 'duckdb'
```

---

### PIT Logic

#### As-Of Date Computation

**Formula**:
```
as_of_base = max(filing_date, period_end + lag_days)
as_of_date = add_trading_days(next_trading_day(as_of_base), extra_trading_lag)
```

Where:
- `lag_days` = 60 for quarterly, 90 for annual
- `extra_trading_lag` = 2 trading days

**Rationale**: Accounts for filing delays and processing/dissemination time.

---

#### Versioning for Restatements

When a restatement occurs (`updated_at` is present):

```
effective_from_version = add_trading_days(next_trading_day(updated_at), extra_trading_lag)
effective_from = max(as_of_date, effective_from_version)
```

**Effect**: Original values remain valid until restatement becomes available.

---

#### Validity Intervals (SCD-2)

For each symbol and statement_type:

```
valid_from = effective_from
valid_to = min(next_effective_from - 1d, next_report_as_of - 1d)
```

Where:
- `next_effective_from`: Next version of same report
- `next_report_as_of`: Next period's as_of_date for same statement_type

**Result**: Non-overlapping intervals defining when each version is valid.

---

#### Interval Join

Using Polars:
```python
joined = daily_pl.join_asof(
    intervals_pl,
    left_on='date',
    right_on='valid_from',
    by='symbol',
    strategy='backward'
).filter(
    pl.col('date') <= pl.col('valid_to')
)
```

Using DuckDB:
```sql
SELECT d.*, f.*, f.effective_from AS source_ts_fund
FROM daily d
LEFT JOIN fundamentals_intervals f
  ON d.symbol = f.symbol
 AND d.date BETWEEN f.valid_from AND f.valid_to
```

---

#### Staleness Handling

If `date - valid_from > stale_max_days`:
- Set fundamental fields to `NaN`
- Set `is_stale_fund = 1`

**Rationale**: Fundamentals older than ~18 months are likely not relevant.

---

### Precedence Policies

#### Quarter Over Annual

**Behavior**:
- Use quarterly data when available and not stale
- Fallback to annual data when quarterly missing/stale
- Results in single set of fundamental columns

**Use Case**: Prefer more frequent updates, fallback to annual.

---

#### Both Suffixes

**Behavior**:
- Expose both quarterly and annual with `_q` and `_y` suffixes
- No automatic override
- Results in two sets of columns

**Use Case**: Let model choose or use both as separate features.

---

### Ratio Computation

#### Filing-Level Ratios

Ratios are computed when filings become available, not daily:

```python
gross_margin = GrossProfit / TotalRevenue
roe = NetIncome / TotalStockholdersEquity
debt_to_equity = TotalDebt / TotalStockholdersEquity
```

**Ratios Computed**:
- **Margins**: `gross_margin`, `operating_margin`, `net_margin`
- **Leverage**: `debt_to_equity`, `debt_to_assets`
- **Coverage**: `interest_coverage`
- **Returns**: `roe`, `roa`, `roic`
- **Accruals**: `accruals_ratio`

---

#### TTM (Trailing Twelve Months)

For quarterly data, TTM ratios sum last 4 quarters:

```python
TotalRevenue_ttm = sum(last 4 quarters of TotalRevenue)
gross_margin_ttm = GrossProfit_ttm / TotalRevenue_ttm
```

**TTM Ratios**: `gross_margin_ttm`, `operating_margin_ttm`, `net_margin_ttm`, `roe_ttm`, `roa_ttm`

---

### Testing

#### Test Suites

1. **test_pit_interval_bounds.py**
   - No fundamentals visible before `valid_from`
   - No fundamentals visible after `valid_to`
   - Boundaries inclusive/exclusive verified
   - Intervals non-overlapping

2. **test_versioning_effect.py**
   - Restatements create multiple versions
   - Values before restatement unchanged
   - `effective_from` after `updated_at`
   - Version IDs ascending

3. **test_precedence_policy.py**
   - Quarter over annual uses quarterly when available
   - Fallback to annual when quarterly missing
   - Both suffixes creates `_q` and `_y` columns
   - No duplicate (symbol, date) pairs

4. **test_leakage_source_ts.py**
   - `source_ts_price <= date` for all rows
   - `source_ts_fund <= date` for all rows
   - `max(source_ts_*) <= date` for all rows

5. **test_staleness.py** (from Quick Wins)
   - Stale fundamentals flagged
   - Stale fundamentals nulled
   - Fresh fundamentals preserved

6. **test_feature_shift.py** & **test_label_alignment.py** (from Quick Wins)
   - Features properly shifted
   - Labels forward-looking
   - No look-ahead bias

---

### Dashboard Diagnostics

Access via Streamlit dashboard:

#### Leakage Table
- Shows rows where `max(source_ts_*) > date`
- Blocks training if violations > 0

#### Coverage Heatmap
- Symbol × date presence visualization
- Shows data availability gaps

#### Fundamentals Age Histogram
- Distribution of `days_since_fund`
- Vertical line at `stale_max_days` threshold

#### Version Counts
- Reports with restatements
- Top restating symbols
- Restatement rate

#### Build Timestamps
- Last build time for each artifact
- Row counts

---

### Usage

#### Build Fundamentals Intervals

```python
from src.fundamentals.pit_processor import PITFundamentalsProcessor

processor = PITFundamentalsProcessor()

# Load raw fundamentals
fundamentals_df = pd.read_parquet('data/raw/fundamentals.parquet')

# Build intervals
intervals = processor.build_fundamentals_intervals(fundamentals_df)

# Save
intervals.to_parquet('data/pit/fundamentals_intervals.parquet')
```

---

#### Join to Daily Panel

```python
# Load daily panel (symbol, date)
daily_df = pd.read_parquet('data/pit/daily_panel_base.parquet')

# Join fundamentals
daily_panel = processor.join_to_daily_panel(daily_df, intervals)

# Validate leakage
validation = processor.validate_leakage(daily_panel)

if not validation['valid']:
    raise ValueError(f"Leakage detected: {validation['total_violations']} violations")

# Save
daily_panel.to_parquet('data/pit/daily_panel.parquet')
```

---

#### Complete Pipeline

```python
# Complete processing
results = processor.process(
    fundamentals_df=fundamentals_df,
    daily_df=daily_df,
    output_dir=Path('data/pit')
)

intervals = results['intervals']
daily_panel = results['daily_panel']
```

---

### Data Dictionary

#### Key Columns

| Column | Type | Description |
|--------|------|-------------|
| `symbol` | str | Ticker symbol (e.g., 'AAPL.US') |
| `date` | date | Trading date |
| `period_end` | date | Financial statement period end date |
| `filing_date` | date | Date when report was filed |
| `updated_at` | datetime | Timestamp of restatement/correction |
| `as_of_date` | date | When original report became available |
| `effective_from` | date | When this version became available |
| `valid_from` | date | Start of validity interval (inclusive) |
| `valid_to` | date | End of validity interval (inclusive) |
| `version_id` | int | Sequential version number (1, 2, 3, ...) |
| `source_ts_price` | datetime | When price data became available |
| `source_ts_fund` | datetime | When fundamental data became available |
| `is_stale_fund` | bool | Whether fundamental is stale (>540 days) |
| `days_since_fund` | int | Days since fundamental became valid |

---

### Best Practices

1. **Always validate leakage** before training:
   ```python
   validation = processor.validate_leakage(daily_panel)
   assert validation['valid'], "Leakage detected!"
   ```

2. **Check staleness** in diagnostics dashboard

3. **Monitor restatement rate** - high rates may indicate data quality issues

4. **Use TTM ratios** for quarterly data to smooth seasonality

5. **Prefer quarter_over_annual** for most use cases

6. **Run tests** before production deployment:
   ```bash
   pytest tests/test_pit_interval_bounds.py -v
   pytest tests/test_versioning_effect.py -v
   pytest tests/test_precedence_policy.py -v
   pytest tests/test_leakage_source_ts.py -v
   ```

---

### Troubleshooting

#### Issue: Leakage violations detected

**Solution**: Check `source_ts_fund` computation in versioning logic. Ensure `effective_from` is computed correctly.

---

#### Issue: Too many stale fundamentals

**Solution**: Adjust `stale_max_days` in config or increase ingestion frequency.

---

#### Issue: Missing fundamentals for some dates

**Solution**: Check coverage heatmap. May need to adjust `lag_days` or `extra_trading_lag`.

---

#### Issue: Restatement rate too high

**Solution**: Verify data source quality. Check for duplicate filings with different `updated_at`.

---

### References

- **Slowly Changing Dimensions (SCD-2)**: https://en.wikipedia.org/wiki/Slowly_changing_dimension
- **Point-in-Time Correctness**: Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
- **Temporal Data Modeling**: Snodgrass, R. (1999). Developing Time-Oriented Database Applications in SQL.

---

*Last Updated: 2024-10-31*
