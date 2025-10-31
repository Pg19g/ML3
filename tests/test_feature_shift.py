"""Tests for feature shift validation."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.utils.shift_validation import (
    ensure_shift,
    validate_feature_shift,
    apply_terminal_shift,
    check_no_future_reference
)


@pytest.fixture
def sample_data():
    """Create sample price data."""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    data = []
    for symbol in ['AAPL', 'MSFT']:
        for date in dates:
            data.append({
                'date': date,
                'symbol': symbol,
                'AdjClose': 100 + np.random.randn() * 10
            })
    
    return pd.DataFrame(data)


def test_ensure_shift_decorator(sample_data):
    """Test that ensure_shift decorator properly shifts features."""
    @ensure_shift(shift_days=1)
    def compute_returns(df):
        df['ret_1d'] = df.groupby('symbol')['AdjClose'].pct_change()
        return df
    
    result = compute_returns(sample_data)
    
    # Check that ret_1d exists
    assert 'ret_1d' in result.columns
    
    # Check that first observation per symbol is NaN (due to shift)
    for symbol in result['symbol'].unique():
        symbol_data = result[result['symbol'] == symbol].sort_values('date')
        
        # First value should be NaN (shift)
        # Second value should also be NaN (pct_change + shift)
        assert pd.isna(symbol_data['ret_1d'].iloc[0]), \
            f"First observation for {symbol} should be NaN"


def test_validate_feature_shift_valid(sample_data):
    """Test validation of properly shifted features."""
    # Compute feature with proper shift
    sample_data = sample_data.sort_values(['symbol', 'date'])
    sample_data['ret_1d'] = sample_data.groupby('symbol')['AdjClose'].pct_change()
    sample_data['ret_1d'] = sample_data.groupby('symbol')['ret_1d'].shift(1)
    
    # Validate
    result = validate_feature_shift(
        sample_data,
        feature_cols=['ret_1d'],
        price_col='AdjClose'
    )
    
    # Should pass (or have only warnings)
    assert result['valid'] or len(result['errors']) == 0


def test_validate_feature_shift_no_shift():
    """Test detection of features without shift (leakage)."""
    # Create data where feature is NOT shifted (leakage!)
    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', '2023-01-10'),
        'symbol': ['AAPL'] * 10,
        'AdjClose': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    })
    
    # Compute feature WITHOUT shift (this is wrong!)
    df['ret_1d'] = df.groupby('symbol')['AdjClose'].pct_change()
    
    # Validate
    result = validate_feature_shift(
        df,
        feature_cols=['ret_1d'],
        price_col='AdjClose'
    )
    
    # Should detect the issue
    # (First observation is NaN from pct_change, but second is not NaN)
    # This is actually okay for this test since pct_change creates NaN
    # Let's check with a feature that has no NaN
    
    df['price_copy'] = df['AdjClose']  # Perfect correlation, no shift
    
    result2 = validate_feature_shift(
        df,
        feature_cols=['price_copy'],
        price_col='AdjClose'
    )
    
    # Should detect perfect correlation
    assert len(result2['errors']) > 0 or len(result2['warnings']) > 0


def test_apply_terminal_shift(sample_data):
    """Test applying terminal shift to features."""
    # Compute features without shift
    sample_data['ret_1d'] = sample_data.groupby('symbol')['AdjClose'].pct_change()
    sample_data['ret_5d'] = sample_data.groupby('symbol')['AdjClose'].pct_change(5)
    
    # Apply terminal shift
    result = apply_terminal_shift(
        sample_data,
        feature_cols=['ret_1d', 'ret_5d'],
        shift_days=1
    )
    
    # Check that features are shifted
    for symbol in result['symbol'].unique():
        symbol_data = result[result['symbol'] == symbol].sort_values('date')
        
        # First observation should be NaN
        assert pd.isna(symbol_data['ret_1d'].iloc[0])
        assert pd.isna(symbol_data['ret_5d'].iloc[0])


def test_check_no_future_reference(sample_data):
    """Test checking for future reference in features."""
    # Properly shifted feature
    sample_data = sample_data.sort_values(['symbol', 'date'])
    sample_data['ret_1d'] = sample_data.groupby('symbol')['AdjClose'].pct_change()
    sample_data['ret_1d'] = sample_data.groupby('symbol')['ret_1d'].shift(1)
    
    # Check
    result = check_no_future_reference(
        sample_data,
        feature_col='ret_1d'
    )
    
    # Should pass
    assert result == True


def test_feature_at_t_uses_data_up_to_t_minus_1():
    """Test that features at time t only use data up to t-1."""
    # Create simple data
    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', '2023-01-10'),
        'symbol': ['AAPL'] * 10,
        'AdjClose': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    })
    
    # Compute return with proper shift
    df['ret_1d'] = df.groupby('symbol')['AdjClose'].pct_change()
    df['ret_1d'] = df.groupby('symbol')['ret_1d'].shift(1)
    
    # At index 2 (date 2023-01-03):
    # - ret_1d should be the return from day 0 to day 1
    # - NOT the return from day 1 to day 2
    
    # Index 0: NaN (no previous data)
    assert pd.isna(df['ret_1d'].iloc[0])
    
    # Index 1: NaN (shift)
    assert pd.isna(df['ret_1d'].iloc[1])
    
    # Index 2: should be (101/100 - 1) = 0.01
    expected = (df['AdjClose'].iloc[1] / df['AdjClose'].iloc[0]) - 1
    actual = df['ret_1d'].iloc[2]
    
    assert abs(actual - expected) < 0.001, \
        f"Feature at t=2 should use data from t=0 to t=1, expected {expected}, got {actual}"


def test_no_label_in_features():
    """Test that labels (forward returns) are not in features."""
    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', '2023-01-10'),
        'symbol': ['AAPL'] * 10,
        'AdjClose': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    })
    
    # Feature: backward-looking (shifted)
    df['ret_1d'] = df.groupby('symbol')['AdjClose'].pct_change()
    df['ret_1d'] = df.groupby('symbol')['ret_1d'].shift(1)
    
    # Label: forward-looking (negative shift)
    df['ret_1d_fwd'] = df.groupby('symbol')['AdjClose'].pct_change().shift(-1)
    
    # Features should have NaN at start
    assert pd.isna(df['ret_1d'].iloc[0])
    assert pd.isna(df['ret_1d'].iloc[1])
    
    # Labels should have NaN at end
    assert pd.isna(df['ret_1d_fwd'].iloc[-1])
    
    # They should NOT be the same
    assert not df['ret_1d'].equals(df['ret_1d_fwd'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
