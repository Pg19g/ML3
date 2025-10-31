"""Tests for schema validation."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.utils.schemas import SchemaValidator, validate_dataframe, enforce_schema


@pytest.fixture
def validator():
    """Create schema validator."""
    return SchemaValidator()


def test_prices_daily_schema_valid(validator):
    """Test valid prices_daily data."""
    df = pd.DataFrame({
        'symbol': ['AAPL.US', 'MSFT.US'],
        'date': [datetime(2023, 1, 3), datetime(2023, 1, 3)],
        'open': [100.0, 200.0],
        'high': [105.0, 210.0],
        'low': [99.0, 198.0],
        'close': [103.0, 205.0],
        'adj_close': [103.0, 205.0],
        'volume': [1000000, 2000000]
    })
    
    result = validator.validate(df, 'prices_daily')
    assert result['valid'], f"Errors: {result['errors']}"


def test_prices_daily_duplicate_pk(validator):
    """Test that duplicate primary keys are detected."""
    df = pd.DataFrame({
        'symbol': ['AAPL.US', 'AAPL.US'],  # Duplicate!
        'date': [datetime(2023, 1, 3), datetime(2023, 1, 3)],  # Duplicate!
        'adj_close': [100.0, 101.0],
        'volume': [1000000, 1000000]
    })
    
    result = validator.validate(df, 'prices_daily')
    assert not result['valid']
    assert any('duplicate' in e.lower() for e in result['errors'])


def test_prices_daily_negative_adj_close(validator):
    """Test that negative adj_close is detected."""
    df = pd.DataFrame({
        'symbol': ['AAPL.US'],
        'date': [datetime(2023, 1, 3)],
        'adj_close': [-100.0],  # Invalid!
        'volume': [1000000]
    })
    
    result = validator.validate(df, 'prices_daily')
    assert not result['valid']
    assert any('adj_close' in e.lower() for e in result['errors'])


def test_prices_daily_negative_volume(validator):
    """Test that negative volume is detected."""
    df = pd.DataFrame({
        'symbol': ['AAPL.US'],
        'date': [datetime(2023, 1, 3)],
        'adj_close': [100.0],
        'volume': [-1000]  # Invalid!
    })
    
    result = validator.validate(df, 'prices_daily')
    assert not result['valid']
    assert any('volume' in e.lower() for e in result['errors'])


def test_prices_daily_high_low_violation(validator):
    """Test that high < low is detected."""
    df = pd.DataFrame({
        'symbol': ['AAPL.US'],
        'date': [datetime(2023, 1, 3)],
        'open': [100.0],
        'high': [95.0],  # Lower than low!
        'low': [99.0],
        'close': [98.0],
        'adj_close': [98.0],
        'volume': [1000000]
    })
    
    result = validator.validate(df, 'prices_daily')
    assert not result['valid']
    assert any('high' in e.lower() and 'low' in e.lower() for e in result['errors'])


def test_fundamentals_schema_valid(validator):
    """Test valid fundamentals data."""
    df = pd.DataFrame({
        'symbol': ['AAPL.US', 'AAPL.US'],
        'statement_type': ['quarterly', 'annual'],
        'period_end': [datetime(2023, 3, 31), datetime(2023, 12, 31)],
        'filing_date': [datetime(2023, 5, 1), datetime(2024, 2, 15)],
        'TotalRevenue': [100e9, 400e9],
        'NetIncome': [20e9, 80e9]
    })
    
    result = validator.validate(df, 'fundamentals')
    assert result['valid'], f"Errors: {result['errors']}"


def test_fundamentals_invalid_statement_type(validator):
    """Test invalid statement_type."""
    df = pd.DataFrame({
        'symbol': ['AAPL.US'],
        'statement_type': ['monthly'],  # Invalid!
        'period_end': [datetime(2023, 3, 31)],
        'TotalRevenue': [100e9]
    })
    
    result = validator.validate(df, 'fundamentals')
    assert not result['valid']
    assert any('statement_type' in e.lower() for e in result['errors'])


def test_fundamentals_filing_before_period_end(validator):
    """Test filing_date < period_end violation."""
    df = pd.DataFrame({
        'symbol': ['AAPL.US'],
        'statement_type': ['quarterly'],
        'period_end': [datetime(2023, 3, 31)],
        'filing_date': [datetime(2023, 3, 1)],  # Before period_end!
        'TotalRevenue': [100e9]
    })
    
    result = validator.validate(df, 'fundamentals')
    assert not result['valid']
    assert any('filing_date' in e.lower() for e in result['errors'])


def test_pit_panel_valid(validator):
    """Test valid PIT panel data."""
    df = pd.DataFrame({
        'symbol': ['AAPL.US'],
        'date': [datetime(2023, 6, 1)],
        'adj_close': [150.0],
        'volume': [1000000],
        'source_ts_price': [datetime(2023, 6, 1, 16, 0)],
        'source_ts_fund': [datetime(2023, 5, 15)],
        'valid_from': [datetime(2023, 5, 15)],
        'valid_to': [datetime(2023, 8, 14)],
        'is_stale_fund': [False]
    })
    
    result = validator.validate(df, 'pit_daily_panel')
    assert result['valid'], f"Errors: {result['errors']}"


def test_pit_panel_source_ts_leakage(validator):
    """Test that source_ts > date is detected (leakage)."""
    df = pd.DataFrame({
        'symbol': ['AAPL.US'],
        'date': [datetime(2023, 6, 1)],
        'adj_close': [150.0],
        'volume': [1000000],
        'source_ts_price': [datetime(2023, 6, 2)],  # Future!
        'source_ts_fund': [datetime(2023, 5, 15)],
        'is_stale_fund': [False]
    })
    
    result = validator.validate(df, 'pit_daily_panel')
    assert not result['valid']
    assert any('leakage' in e.lower() or 'source_ts_price' in e.lower() 
               for e in result['errors'])


def test_pit_panel_valid_from_leakage(validator):
    """Test that valid_from > date is detected."""
    df = pd.DataFrame({
        'symbol': ['AAPL.US'],
        'date': [datetime(2023, 6, 1)],
        'adj_close': [150.0],
        'volume': [1000000],
        'source_ts_price': [datetime(2023, 6, 1)],
        'source_ts_fund': [datetime(2023, 5, 15)],
        'valid_from': [datetime(2023, 6, 15)],  # Future!
        'is_stale_fund': [False]
    })
    
    result = validator.validate(df, 'pit_daily_panel')
    assert not result['valid']
    assert any('valid_from' in e.lower() for e in result['errors'])


def test_features_schema_valid(validator):
    """Test valid features data."""
    df = pd.DataFrame({
        'symbol': ['AAPL.US'],
        'date': [datetime(2023, 6, 1)],
        'ret_1d': [0.01],
        'vol_20d': [0.25],
        'net_margin': [0.20],
        'source_ts_price': [datetime(2023, 6, 1)],
        'source_ts_fund': [datetime(2023, 5, 15)]
    })
    
    result = validator.validate(df, 'features')
    assert result['valid'], f"Errors: {result['errors']}"


def test_labels_schema_valid(validator):
    """Test valid labels data."""
    df = pd.DataFrame({
        'symbol': ['AAPL.US'],
        'date': [datetime(2023, 6, 1)],
        'ret_1d_fwd': [0.02],
        'ret_5d_fwd': [0.05],
        'ret_21d_fwd': [0.10]
    })
    
    result = validator.validate(df, 'labels')
    assert result['valid'], f"Errors: {result['errors']}"


def test_enforce_schema_raises():
    """Test that enforce_schema raises on errors."""
    df = pd.DataFrame({
        'symbol': ['AAPL.US', 'AAPL.US'],  # Duplicate
        'date': [datetime(2023, 1, 3), datetime(2023, 1, 3)],
        'adj_close': [100.0, 100.0]
    })
    
    with pytest.raises(ValueError, match="validation failed"):
        enforce_schema(df, 'prices_daily', raise_on_error=True)


def test_enforce_schema_no_raise():
    """Test that enforce_schema doesn't raise when flag is False."""
    df = pd.DataFrame({
        'symbol': ['AAPL.US', 'AAPL.US'],  # Duplicate
        'date': [datetime(2023, 1, 3), datetime(2023, 1, 3)],
        'adj_close': [100.0, 100.0]
    })
    
    # Should not raise
    result = enforce_schema(df, 'prices_daily', raise_on_error=False)
    assert result is df


def test_get_schema_info(validator):
    """Test getting schema information."""
    info = validator.get_schema_info('prices_daily')
    
    assert 'description' in info
    assert info['primary_key'] == ['symbol', 'date']
    assert 'symbol' in info['columns']
    assert 'adj_close' in info['columns']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
