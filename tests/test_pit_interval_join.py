"""Tests for PIT interval join logic."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.pit_enhanced import PITProcessorEnhanced


@pytest.fixture
def processor():
    """Create enhanced PIT processor."""
    return PITProcessorEnhanced()


@pytest.fixture
def sample_fundamentals():
    """Create sample fundamental data with multiple reports."""
    return pd.DataFrame({
        'symbol': ['AAPL'] * 3,
        'statement_type': ['quarterly', 'quarterly', 'quarterly'],
        'period_end': [
            datetime(2023, 3, 31),
            datetime(2023, 6, 30),
            datetime(2023, 9, 30)
        ],
        'filing_date': [None, None, None],
        'TotalRevenue': [100e9, 110e9, 120e9],
        'NetIncome': [20e9, 22e9, 24e9]
    })


@pytest.fixture
def sample_prices():
    """Create sample daily price data."""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    # Filter to trading days (Mon-Fri, simplified)
    dates = [d for d in dates if d.weekday() < 5]
    
    return pd.DataFrame({
        'date': dates,
        'symbol': ['AAPL'] * len(dates),
        'adj_close': 150.0 + np.random.randn(len(dates)).cumsum(),
        'volume': np.random.randint(1000000, 10000000, len(dates))
    })


def test_validity_intervals_no_overlap(processor, sample_fundamentals):
    """Test that validity intervals don't overlap."""
    # Compute as_of_date
    fund_with_asof = processor.compute_as_of_date(sample_fundamentals)
    
    # Build validity intervals
    fund_with_validity = processor.build_validity_intervals(fund_with_asof)
    
    # Check no overlap
    for i in range(len(fund_with_validity) - 1):
        current_valid_to = fund_with_validity.iloc[i]['valid_to']
        next_valid_from = fund_with_validity.iloc[i + 1]['valid_from']
        
        # valid_to should be < next valid_from
        assert current_valid_to < next_valid_from, \
            f"Overlap detected: valid_to {current_valid_to} >= next valid_from {next_valid_from}"


def test_no_fundamentals_before_valid_from(processor, sample_fundamentals, sample_prices):
    """Test that no fundamentals are visible before valid_from."""
    # Build PIT panel
    pit_panel = processor.build_pit_panel(
        sample_prices,
        sample_fundamentals,
        fundamental_cols=['TotalRevenue', 'NetIncome']
    )
    
    # Check that for all rows with fundamentals, date >= valid_from
    has_fund = pit_panel['TotalRevenue'].notna()
    
    if has_fund.any():
        fund_rows = pit_panel[has_fund]
        invalid = fund_rows['date'] < fund_rows['valid_from']
        
        assert not invalid.any(), \
            f"Found {invalid.sum()} rows with fundamentals before valid_from"


def test_no_fundamentals_after_valid_to(processor, sample_fundamentals, sample_prices):
    """Test that no fundamentals are visible after valid_to."""
    # Build PIT panel
    pit_panel = processor.build_pit_panel(
        sample_prices,
        sample_fundamentals,
        fundamental_cols=['TotalRevenue', 'NetIncome']
    )
    
    # Check that for all rows with fundamentals, date <= valid_to
    has_fund = pit_panel['TotalRevenue'].notna()
    
    if has_fund.any():
        fund_rows = pit_panel[has_fund]
        invalid = fund_rows['date'] > fund_rows['valid_to']
        
        assert not invalid.any(), \
            f"Found {invalid.sum()} rows with fundamentals after valid_to"


def test_as_of_behavior_on_boundary(processor):
    """Test as-of join behavior on boundary dates."""
    # Create fundamentals with known as_of_dates
    fundamentals = pd.DataFrame({
        'symbol': ['AAPL', 'AAPL'],
        'statement_type': ['quarterly', 'quarterly'],
        'period_end': [datetime(2023, 3, 31), datetime(2023, 6, 30)],
        'filing_date': [None, None],
        'TotalRevenue': [100e9, 110e9]
    })
    
    # Compute as_of_date
    fund_with_asof = processor.compute_as_of_date(fundamentals)
    fund_with_validity = processor.build_validity_intervals(fund_with_asof)
    
    # Get the exact valid_from dates
    first_valid_from = fund_with_validity.iloc[0]['valid_from']
    second_valid_from = fund_with_validity.iloc[1]['valid_from']
    
    # Create prices around these dates
    prices = pd.DataFrame({
        'date': [
            first_valid_from - timedelta(days=1),  # Before first report
            first_valid_from,                       # Exactly when first report available
            second_valid_from - timedelta(days=1),  # Day before second report
            second_valid_from,                       # Exactly when second report available
        ],
        'symbol': ['AAPL'] * 4,
        'adj_close': [150.0] * 4,
        'volume': [1000000] * 4
    })
    
    # Build PIT panel
    pit_panel = processor.build_pit_panel(
        prices,
        fundamentals,
        fundamental_cols=['TotalRevenue']
    )
    
    # Check behavior
    # Day before first report: no fundamentals
    before_first = pit_panel[pit_panel['date'] == first_valid_from - timedelta(days=1)]
    if len(before_first) > 0:
        assert before_first['TotalRevenue'].isna().all(), \
            "Fundamentals should not be available before valid_from"
    
    # Exactly on first valid_from: first report available
    on_first = pit_panel[pit_panel['date'] == first_valid_from]
    if len(on_first) > 0:
        assert on_first['TotalRevenue'].iloc[0] == 100e9, \
            "First report should be available on valid_from"
    
    # Day before second report: still first report
    before_second = pit_panel[pit_panel['date'] == second_valid_from - timedelta(days=1)]
    if len(before_second) > 0:
        assert before_second['TotalRevenue'].iloc[0] == 100e9, \
            "First report should still be active before second valid_from"
    
    # Exactly on second valid_from: second report available
    on_second = pit_panel[pit_panel['date'] == second_valid_from]
    if len(on_second) > 0:
        assert on_second['TotalRevenue'].iloc[0] == 110e9, \
            "Second report should be available on its valid_from"


def test_source_timestamps_present(processor, sample_fundamentals, sample_prices):
    """Test that source timestamps are added."""
    pit_panel = processor.build_pit_panel(
        sample_prices,
        sample_fundamentals,
        fundamental_cols=['TotalRevenue', 'NetIncome']
    )
    
    # Check that source timestamp columns exist
    assert 'source_ts_price' in pit_panel.columns
    assert 'source_ts_fund' in pit_panel.columns
    
    # Check that source_ts_price is not null
    assert pit_panel['source_ts_price'].notna().all()


def test_source_timestamps_no_leakage(processor, sample_fundamentals, sample_prices):
    """Test that source timestamps don't leak."""
    pit_panel = processor.build_pit_panel(
        sample_prices,
        sample_fundamentals,
        fundamental_cols=['TotalRevenue', 'NetIncome']
    )
    
    # Check source_ts_price <= date
    pit_panel['ts_price_date'] = pd.to_datetime(pit_panel['source_ts_price']).dt.date
    pit_panel['date_only'] = pd.to_datetime(pit_panel['date']).dt.date
    
    price_leakage = pit_panel['ts_price_date'] > pit_panel['date_only']
    assert not price_leakage.any(), \
        f"Price timestamp leakage detected in {price_leakage.sum()} rows"
    
    # Check source_ts_fund <= date (where not null)
    has_fund = pit_panel['source_ts_fund'].notna()
    if has_fund.any():
        pit_panel['ts_fund_date'] = pd.to_datetime(pit_panel['source_ts_fund']).dt.date
        
        fund_leakage = has_fund & (pit_panel['ts_fund_date'] > pit_panel['date_only'])
        assert not fund_leakage.any(), \
            f"Fundamental timestamp leakage detected in {fund_leakage.sum()} rows"


def test_pit_integrity_validation(processor, sample_fundamentals, sample_prices):
    """Test PIT integrity validation."""
    pit_panel = processor.build_pit_panel(
        sample_prices,
        sample_fundamentals,
        fundamental_cols=['TotalRevenue', 'NetIncome']
    )
    
    # Run validation
    result = processor.validate_pit_integrity(pit_panel)
    
    # Should pass
    assert result['passed'], \
        f"PIT integrity check failed with {result['total_violations']} violations"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
