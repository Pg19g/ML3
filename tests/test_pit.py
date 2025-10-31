"""Tests for point-in-time logic."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.pit import PITProcessor


@pytest.fixture
def sample_fundamentals():
    """Create sample fundamentals data."""
    data = {
        'symbol': ['AAPL', 'AAPL', 'AAPL'],
        'period_end': [
            datetime(2023, 3, 31),
            datetime(2023, 6, 30),
            datetime(2023, 9, 30)
        ],
        'statement_type': ['quarterly', 'quarterly', 'quarterly'],
        'filing_date': [None, None, None],
        'revenue': [100, 110, 120]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_prices():
    """Create sample price data."""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    data = {
        'date': dates,
        'symbol': ['AAPL'] * len(dates),
        'AdjClose': np.random.randn(len(dates)).cumsum() + 100
    }
    return pd.DataFrame(data)


def test_compute_as_of_date(sample_fundamentals):
    """Test as_of_date computation."""
    processor = PITProcessor()
    
    result = processor.compute_as_of_date(sample_fundamentals)
    
    assert 'as_of_date' in result.columns
    assert all(result['as_of_date'] > result['period_end'])
    
    # Check lag is applied
    for idx, row in result.iterrows():
        days_diff = (row['as_of_date'] - row['period_end']).days
        assert days_diff >= 60  # At least q_lag_days


def test_compute_validity_intervals(sample_fundamentals):
    """Test validity interval computation."""
    processor = PITProcessor()
    
    # First compute as_of_date
    with_asof = processor.compute_as_of_date(sample_fundamentals)
    
    # Then compute validity intervals
    result = processor.compute_validity_intervals(with_asof)
    
    assert 'valid_from' in result.columns
    assert 'valid_to' in result.columns
    
    # Check intervals don't overlap
    for i in range(len(result) - 1):
        assert result.iloc[i]['valid_to'] < result.iloc[i + 1]['valid_from']


def test_pit_integrity_check():
    """Test PIT integrity check."""
    processor = PITProcessor()
    
    # Create valid data
    valid_data = pd.DataFrame({
        'date': [datetime(2023, 6, 1), datetime(2023, 7, 1)],
        'symbol': ['AAPL', 'AAPL'],
        'valid_from': [datetime(2023, 5, 1), datetime(2023, 6, 1)],
        'period_end': [datetime(2023, 3, 31), datetime(2023, 6, 30)]
    })
    
    result = processor.check_pit_integrity(valid_data)
    assert result['passed'] is True
    assert result['violations'] == 0
    
    # Create invalid data (future leakage)
    invalid_data = pd.DataFrame({
        'date': [datetime(2023, 6, 1)],
        'symbol': ['AAPL'],
        'valid_from': [datetime(2023, 7, 1)],  # Future date!
        'period_end': [datetime(2023, 3, 31)]
    })
    
    result = processor.check_pit_integrity(invalid_data)
    assert result['passed'] is False
    assert result['violations'] > 0


def test_as_of_join():
    """Test as-of join logic."""
    processor = PITProcessor()
    
    # Daily panel
    daily_panel = pd.DataFrame({
        'date': pd.date_range('2023-06-01', '2023-06-10'),
        'symbol': ['AAPL'] * 10
    })
    
    # Fundamentals with validity
    fundamentals = pd.DataFrame({
        'symbol': ['AAPL', 'AAPL'],
        'valid_from': [datetime(2023, 6, 1), datetime(2023, 6, 6)],
        'valid_to': [datetime(2023, 6, 5), datetime(2023, 6, 30)],
        'revenue': [100, 110]
    })
    
    result = processor.as_of_join(daily_panel, fundamentals)
    
    # Check that each date gets the correct fundamental
    assert len(result) == 10
    
    # Dates 1-5 should have revenue=100
    early_dates = result[result['date'] <= datetime(2023, 6, 5)]
    assert all(early_dates['revenue'] == 100)
    
    # Dates 6-10 should have revenue=110
    late_dates = result[result['date'] >= datetime(2023, 6, 6)]
    assert all(late_dates['revenue'] == 110)


if __name__ == '__main__':
    pytest.main([__file__])
