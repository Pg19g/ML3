"""Tests for source timestamp leakage detection."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.fundamentals.pit_processor import PITFundamentalsProcessor


@pytest.fixture
def sample_daily_panel():
    """Create sample daily panel."""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    return pd.DataFrame({
        'symbol': ['AAPL.US'] * len(dates),
        'date': dates,
        'close': 100.0 + np.random.randn(len(dates)).cumsum()
    })


@pytest.fixture
def sample_fundamentals():
    """Create sample fundamentals."""
    return pd.DataFrame({
        'symbol': ['AAPL.US'] * 4,
        'statement_type': ['quarterly'] * 4,
        'period_end': pd.to_datetime([
            '2023-03-31', '2023-06-30', '2023-09-30', '2023-12-31'
        ]),
        'filing_date': pd.to_datetime([
            '2023-05-01', '2023-08-01', '2023-11-01', '2024-02-01'
        ]),
        'TotalRevenue': [100, 110, 120, 130],
        'NetIncome': [20, 22, 24, 26]
    })


def test_source_ts_price_not_after_date(sample_daily_panel, sample_fundamentals):
    """Test that source_ts_price <= date for all rows."""
    processor = PITFundamentalsProcessor()
    
    # Build intervals
    intervals = processor.build_fundamentals_intervals(sample_fundamentals)
    
    # Join to daily panel
    daily_panel = processor.join_to_daily_panel(sample_daily_panel, intervals)
    
    # Validate
    validation = processor.validate_leakage(daily_panel)
    
    # Should have no price timestamp violations
    price_violations = [
        v for v in validation['violations']
        if v['type'] == 'source_ts_price > date'
    ]
    
    assert len(price_violations) == 0, \
        f"Found {len(price_violations)} price timestamp violations"


def test_source_ts_fund_not_after_date(sample_daily_panel, sample_fundamentals):
    """Test that source_ts_fund <= date for all rows."""
    processor = PITFundamentalsProcessor()
    
    # Build intervals
    intervals = processor.build_fundamentals_intervals(sample_fundamentals)
    
    # Join to daily panel
    daily_panel = processor.join_to_daily_panel(sample_daily_panel, intervals)
    
    # Validate
    validation = processor.validate_leakage(daily_panel)
    
    # Should have no fundamental timestamp violations
    fund_violations = [
        v for v in validation['violations']
        if v['type'] == 'source_ts_fund > date'
    ]
    
    assert len(fund_violations) == 0, \
        f"Found {len(fund_violations)} fundamental timestamp violations"


def test_leakage_validation_passes(sample_daily_panel, sample_fundamentals):
    """Test that leakage validation passes for valid data."""
    processor = PITFundamentalsProcessor()
    
    # Build intervals
    intervals = processor.build_fundamentals_intervals(sample_fundamentals)
    
    # Join to daily panel
    daily_panel = processor.join_to_daily_panel(sample_daily_panel, intervals)
    
    # Validate
    validation = processor.validate_leakage(daily_panel)
    
    assert validation['valid'], \
        f"Leakage validation failed: {validation['violations']}"
    
    assert validation['total_violations'] == 0, \
        f"Found {validation['total_violations']} violations"


def test_detect_future_price_timestamp():
    """Test detection of future price timestamps."""
    # Create data with future price timestamp (should fail)
    df = pd.DataFrame({
        'symbol': ['AAPL.US'],
        'date': pd.to_datetime(['2023-01-01']),
        'source_ts_price': pd.to_datetime(['2023-01-02']),  # Future!
        'source_ts_fund': pd.to_datetime(['2022-12-31'])
    })
    
    processor = PITFundamentalsProcessor()
    validation = processor.validate_leakage(df)
    
    assert not validation['valid'], \
        "Should detect future price timestamp"
    
    assert validation['total_violations'] > 0, \
        "Should have violations"


def test_detect_future_fund_timestamp():
    """Test detection of future fundamental timestamps."""
    # Create data with future fundamental timestamp (should fail)
    df = pd.DataFrame({
        'symbol': ['AAPL.US'],
        'date': pd.to_datetime(['2023-01-01']),
        'source_ts_price': pd.to_datetime(['2022-12-31']),
        'source_ts_fund': pd.to_datetime(['2023-01-02'])  # Future!
    })
    
    processor = PITFundamentalsProcessor()
    validation = processor.validate_leakage(df)
    
    assert not validation['valid'], \
        "Should detect future fundamental timestamp"
    
    assert validation['total_violations'] > 0, \
        "Should have violations"


def test_max_source_ts_not_after_date():
    """Test that max(source_ts_price, source_ts_fund) <= date."""
    df = pd.DataFrame({
        'symbol': ['AAPL.US'] * 3,
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'source_ts_price': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'source_ts_fund': pd.to_datetime(['2022-12-31', '2023-01-01', '2023-01-02'])
    })
    
    # Compute max
    df['max_source_ts'] = df[['source_ts_price', 'source_ts_fund']].max(axis=1)
    
    # Check max <= date
    violations = df['max_source_ts'] > df['date']
    
    assert not violations.any(), \
        f"Found {violations.sum()} rows where max(source_ts_*) > date"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
