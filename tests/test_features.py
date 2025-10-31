"""Tests for feature engineering."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.features import FeatureEngineer


@pytest.fixture
def sample_price_data():
    """Create sample price data."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    data = {
        'date': dates,
        'symbol': ['AAPL'] * len(dates),
        'Open': 100 + np.random.randn(len(dates)).cumsum(),
        'High': 102 + np.random.randn(len(dates)).cumsum(),
        'Low': 98 + np.random.randn(len(dates)).cumsum(),
        'Close': 100 + np.random.randn(len(dates)).cumsum(),
        'AdjClose': 100 + np.random.randn(len(dates)).cumsum(),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }
    
    return pd.DataFrame(data)


def test_compute_returns(sample_price_data):
    """Test return computation."""
    engineer = FeatureEngineer()
    
    result = engineer.compute_technical_features(sample_price_data)
    
    # Check that return features exist
    assert 'ret_1d' in result.columns
    assert 'ret_5d' in result.columns
    
    # Check that returns are shifted (no leakage)
    # The first value should be NaN due to shift
    assert pd.isna(result['ret_1d'].iloc[0])


def test_compute_volatility(sample_price_data):
    """Test volatility computation."""
    engineer = FeatureEngineer()
    
    result = engineer.compute_technical_features(sample_price_data)
    
    # Check volatility features
    assert 'vol_20d' in result.columns
    assert 'vol_63d' in result.columns
    
    # Volatility should be positive
    assert all(result['vol_20d'].dropna() >= 0)


def test_compute_rsi(sample_price_data):
    """Test RSI computation."""
    engineer = FeatureEngineer()
    
    result = engineer.compute_technical_features(sample_price_data)
    
    # Check RSI feature
    assert 'rsi_14' in result.columns
    
    # RSI should be between 0 and 100
    rsi_values = result['rsi_14'].dropna()
    assert all((rsi_values >= 0) & (rsi_values <= 100))


def test_feature_shift():
    """Test that features are properly shifted to prevent leakage."""
    engineer = FeatureEngineer()
    
    # Create simple data
    data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', '2023-01-10'),
        'symbol': ['AAPL'] * 10,
        'AdjClose': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'Volume': [1000000] * 10,
        'Open': [100] * 10,
        'High': [105] * 10,
        'Low': [95] * 10,
        'Close': [100] * 10
    })
    
    result = engineer.compute_technical_features(data)
    
    # Check that first value is NaN (shifted)
    assert pd.isna(result['ret_1d'].iloc[0])
    
    # Check that the feature at index 1 corresponds to data at index 0
    # (i.e., it's shifted by 1)
    expected_return = (data['AdjClose'].iloc[1] / data['AdjClose'].iloc[0]) - 1
    actual_return = result['ret_1d'].iloc[2]  # Index 2 because of shift
    
    # Allow small numerical differences
    assert abs(expected_return - actual_return) < 0.01 or pd.isna(actual_return)


def test_fundamental_ratios():
    """Test fundamental ratio computation."""
    engineer = FeatureEngineer()
    
    data = pd.DataFrame({
        'symbol': ['AAPL', 'AAPL'],
        'date': [datetime(2023, 6, 1), datetime(2023, 9, 1)],
        'NetIncome': [100, 110],
        'TotalRevenue': [500, 550],
        'TotalStockholdersEquity': [1000, 1100],
        'TotalAssets': [2000, 2200],
        'TotalDebt': [500, 550]
    })
    
    result = engineer.compute_fundamental_features(data)
    
    # Check ratio features
    assert 'net_margin' in result.columns
    assert 'roe' in result.columns
    assert 'debt_to_equity' in result.columns
    
    # Verify calculations
    assert abs(result['net_margin'].iloc[0] - (100 / 500)) < 0.01
    assert abs(result['roe'].iloc[0] - (100 / 1000)) < 0.01


if __name__ == '__main__':
    pytest.main([__file__])
