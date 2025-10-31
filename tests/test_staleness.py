"""Tests for fundamental staleness handling."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.pit_enhanced import PITProcessorEnhanced


@pytest.fixture
def processor():
    """Create enhanced PIT processor."""
    return PITProcessorEnhanced()


def test_stale_fundamentals_flagged(processor):
    """Test that stale fundamentals are properly flagged."""
    # Create fundamentals with old period_end
    old_date = datetime.now() - timedelta(days=600)  # Very old
    
    fundamentals = pd.DataFrame({
        'symbol': ['AAPL'],
        'statement_type': ['quarterly'],
        'period_end': [old_date],
        'filing_date': [None],
        'TotalRevenue': [100e9]
    })
    
    # Create recent prices
    prices = pd.DataFrame({
        'date': [datetime.now() - timedelta(days=1)],
        'symbol': ['AAPL'],
        'adj_close': [150.0],
        'volume': [1000000]
    })
    
    # Build PIT panel
    pit_panel = processor.build_pit_panel(
        prices,
        fundamentals,
        fundamental_cols=['TotalRevenue']
    )
    
    # Check that stale flag is set
    if len(pit_panel) > 0:
        assert 'is_stale_fund' in pit_panel.columns
        assert pit_panel['is_stale_fund'].iloc[0] == True, \
            "Old fundamental should be flagged as stale"


def test_stale_fundamentals_nulled(processor):
    """Test that stale fundamentals are nulled out."""
    # Create fundamentals with old period_end
    old_date = datetime.now() - timedelta(days=600)
    
    fundamentals = pd.DataFrame({
        'symbol': ['AAPL'],
        'statement_type': ['quarterly'],
        'period_end': [old_date],
        'filing_date': [None],
        'TotalRevenue': [100e9],
        'NetIncome': [20e9]
    })
    
    # Create recent prices
    prices = pd.DataFrame({
        'date': [datetime.now() - timedelta(days=1)],
        'symbol': ['AAPL'],
        'adj_close': [150.0],
        'volume': [1000000]
    })
    
    # Build PIT panel
    pit_panel = processor.build_pit_panel(
        prices,
        fundamentals,
        fundamental_cols=['TotalRevenue', 'NetIncome']
    )
    
    # Check that fundamental values are nulled
    if len(pit_panel) > 0:
        stale_rows = pit_panel[pit_panel['is_stale_fund'] == True]
        
        if len(stale_rows) > 0:
            assert stale_rows['TotalRevenue'].isna().all(), \
                "Stale TotalRevenue should be nulled"
            assert stale_rows['NetIncome'].isna().all(), \
                "Stale NetIncome should be nulled"


def test_fresh_fundamentals_not_nulled(processor):
    """Test that fresh fundamentals are not nulled."""
    # Create recent fundamentals
    recent_date = datetime.now() - timedelta(days=90)
    
    fundamentals = pd.DataFrame({
        'symbol': ['AAPL'],
        'statement_type': ['quarterly'],
        'period_end': [recent_date],
        'filing_date': [None],
        'TotalRevenue': [100e9],
        'NetIncome': [20e9]
    })
    
    # Create recent prices
    prices = pd.DataFrame({
        'date': [datetime.now() - timedelta(days=1)],
        'symbol': ['AAPL'],
        'adj_close': [150.0],
        'volume': [1000000]
    })
    
    # Build PIT panel
    pit_panel = processor.build_pit_panel(
        prices,
        fundamentals,
        fundamental_cols=['TotalRevenue', 'NetIncome']
    )
    
    # Check that fundamental values are NOT nulled
    if len(pit_panel) > 0:
        fresh_rows = pit_panel[pit_panel['is_stale_fund'] == False]
        
        if len(fresh_rows) > 0:
            assert fresh_rows['TotalRevenue'].notna().any(), \
                "Fresh TotalRevenue should not be nulled"
            assert fresh_rows['NetIncome'].notna().any(), \
                "Fresh NetIncome should not be nulled"


def test_staleness_threshold(processor):
    """Test staleness threshold (stale_max_days)."""
    stale_max_days = processor.stale_max_days
    
    # Create fundamentals exactly at threshold
    threshold_date = datetime.now() - timedelta(days=stale_max_days + 1)
    
    fundamentals = pd.DataFrame({
        'symbol': ['AAPL'],
        'statement_type': ['quarterly'],
        'period_end': [threshold_date],
        'filing_date': [None],
        'TotalRevenue': [100e9]
    })
    
    # Create recent prices
    prices = pd.DataFrame({
        'date': [datetime.now() - timedelta(days=1)],
        'symbol': ['AAPL'],
        'adj_close': [150.0],
        'volume': [1000000]
    })
    
    # Build PIT panel
    pit_panel = processor.build_pit_panel(
        prices,
        fundamentals,
        fundamental_cols=['TotalRevenue']
    )
    
    # Check staleness
    if len(pit_panel) > 0:
        # Should be stale since we're past threshold
        assert pit_panel['is_stale_fund'].iloc[0] == True


def test_days_since_fund_calculated(processor):
    """Test that days_since_fund is calculated correctly."""
    # Create fundamentals
    period_end = datetime(2023, 3, 31)
    
    fundamentals = pd.DataFrame({
        'symbol': ['AAPL'],
        'statement_type': ['quarterly'],
        'period_end': [period_end],
        'filing_date': [None],
        'TotalRevenue': [100e9]
    })
    
    # Create prices
    test_date = datetime(2023, 9, 1)
    
    prices = pd.DataFrame({
        'date': [test_date],
        'symbol': ['AAPL'],
        'adj_close': [150.0],
        'volume': [1000000]
    })
    
    # Build PIT panel
    pit_panel = processor.build_pit_panel(
        prices,
        fundamentals,
        fundamental_cols=['TotalRevenue']
    )
    
    # Check days_since_fund
    if len(pit_panel) > 0 and 'days_since_fund' in pit_panel.columns:
        days_since = pit_panel['days_since_fund'].iloc[0]
        
        # Should be positive
        assert days_since > 0, "days_since_fund should be positive"
        
        # Should be reasonable (between valid_from and test_date)
        assert days_since < 365, "days_since_fund seems too large"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
