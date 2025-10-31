"""Tests for label alignment with features."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.utils.shift_validation import validate_label_alignment


@pytest.fixture
def sample_features():
    """Create sample features data."""
    dates = pd.date_range('2023-01-01', '2023-01-31', freq='D')
    
    data = []
    for symbol in ['AAPL', 'MSFT']:
        for date in dates:
            data.append({
                'date': date,
                'symbol': symbol,
                'ret_1d': np.random.randn() * 0.01,
                'vol_20d': np.random.rand() * 0.3
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_labels():
    """Create sample labels data."""
    dates = pd.date_range('2023-01-01', '2023-01-31', freq='D')
    
    data = []
    for symbol in ['AAPL', 'MSFT']:
        for date in dates:
            data.append({
                'date': date,
                'symbol': symbol,
                'ret_1d_fwd': np.random.randn() * 0.01,
                'ret_5d_fwd': np.random.randn() * 0.03
            })
    
    return pd.DataFrame(data)


def test_perfect_alignment(sample_features, sample_labels):
    """Test validation when features and labels are perfectly aligned."""
    result = validate_label_alignment(
        sample_features,
        sample_labels,
        key_cols=['symbol', 'date']
    )
    
    # Should be aligned
    assert result['aligned']
    assert result['overlap'] == result['total']


def test_misaligned_missing_labels():
    """Test detection of missing labels."""
    features = pd.DataFrame({
        'date': pd.date_range('2023-01-01', '2023-01-10'),
        'symbol': ['AAPL'] * 10,
        'ret_1d': np.random.randn(10)
    })
    
    # Labels missing last 2 days
    labels = pd.DataFrame({
        'date': pd.date_range('2023-01-01', '2023-01-08'),
        'symbol': ['AAPL'] * 8,
        'ret_1d_fwd': np.random.randn(8)
    })
    
    result = validate_label_alignment(features, labels)
    
    # Should detect mismatch
    assert len(result['warnings']) > 0 or len(result['errors']) > 0


def test_misaligned_extra_labels():
    """Test detection of extra labels."""
    features = pd.DataFrame({
        'date': pd.date_range('2023-01-01', '2023-01-08'),
        'symbol': ['AAPL'] * 8,
        'ret_1d': np.random.randn(8)
    })
    
    # Labels have extra days
    labels = pd.DataFrame({
        'date': pd.date_range('2023-01-01', '2023-01-10'),
        'symbol': ['AAPL'] * 10,
        'ret_1d_fwd': np.random.randn(10)
    })
    
    result = validate_label_alignment(features, labels)
    
    # Should detect mismatch
    assert len(result['warnings']) > 0 or len(result['errors']) > 0


def test_forward_looking_labels():
    """Test that labels are forward-looking (NaN at end)."""
    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', '2023-01-10'),
        'symbol': ['AAPL'] * 10,
        'AdjClose': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    })
    
    # Compute forward returns (negative shift)
    df['ret_1d_fwd'] = df.groupby('symbol')['AdjClose'].pct_change().shift(-1)
    df['ret_5d_fwd'] = df.groupby('symbol')['AdjClose'].pct_change(5).shift(-5)
    
    # Last observation should be NaN
    assert pd.isna(df['ret_1d_fwd'].iloc[-1])
    assert pd.isna(df['ret_5d_fwd'].iloc[-1])
    
    # First observation should NOT be NaN (forward-looking)
    assert pd.notna(df['ret_1d_fwd'].iloc[0])


def test_features_and_labels_share_index():
    """Test that features and labels share the same (symbol, date) index."""
    dates = pd.date_range('2023-01-01', '2023-01-10')
    
    features = pd.DataFrame({
        'date': dates,
        'symbol': ['AAPL'] * 10,
        'ret_1d': np.random.randn(10)
    })
    
    labels = pd.DataFrame({
        'date': dates,
        'symbol': ['AAPL'] * 10,
        'ret_1d_fwd': np.random.randn(10)
    })
    
    # Merge should work perfectly
    merged = features.merge(labels, on=['symbol', 'date'], how='inner')
    
    assert len(merged) == len(features)
    assert len(merged) == len(labels)


def test_expected_offsets():
    """Test that features and labels have expected temporal offsets."""
    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', '2023-01-10'),
        'symbol': ['AAPL'] * 10,
        'AdjClose': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    })
    
    # Feature: uses data up to t-1
    df['ret_1d'] = df.groupby('symbol')['AdjClose'].pct_change()
    df['ret_1d'] = df.groupby('symbol')['ret_1d'].shift(1)
    
    # Label: predicts t+1
    df['ret_1d_fwd'] = df.groupby('symbol')['AdjClose'].pct_change().shift(-1)
    
    # At index 2:
    # - ret_1d uses data from index 0 to 1
    # - ret_1d_fwd predicts from index 2 to 3
    
    # Check index 2
    feature_val = df['ret_1d'].iloc[2]
    label_val = df['ret_1d_fwd'].iloc[2]
    
    # Feature should be return from 0 to 1
    expected_feature = (df['AdjClose'].iloc[1] / df['AdjClose'].iloc[0]) - 1
    
    # Label should be return from 2 to 3
    expected_label = (df['AdjClose'].iloc[3] / df['AdjClose'].iloc[2]) - 1
    
    assert abs(feature_val - expected_feature) < 0.001
    assert abs(label_val - expected_label) < 0.001
    
    # They should be different
    assert feature_val != label_val


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
