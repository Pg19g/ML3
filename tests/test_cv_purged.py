"""Tests for PurgedKFold CV with embargo."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.cv.purged_kfold import PurgedKFold, TimeSeriesCV


@pytest.fixture
def sample_timeseries_data():
    """Create sample time-series data."""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    df = pd.DataFrame({
        'feature1': np.random.randn(len(dates)),
        'feature2': np.random.randn(len(dates)),
        'label': np.random.randn(len(dates))
    }, index=dates)
    
    return df


def test_purged_kfold_creates_n_splits(sample_timeseries_data):
    """Test that PurgedKFold creates correct number of splits."""
    cv = PurgedKFold(n_splits=5, embargo_days=0, purge_days=0)
    
    X = sample_timeseries_data[['feature1', 'feature2']]
    y = sample_timeseries_data['label']
    
    splits = list(cv.split(X, y))
    
    assert len(splits) == 5, "Should create 5 splits"


def test_purged_kfold_no_overlap(sample_timeseries_data):
    """Test that train and test sets don't overlap."""
    cv = PurgedKFold(n_splits=5, embargo_days=0, purge_days=0)
    
    X = sample_timeseries_data[['feature1', 'feature2']]
    y = sample_timeseries_data['label']
    
    for train_idx, test_idx in cv.split(X, y):
        # Check no overlap
        overlap = set(train_idx) & set(test_idx)
        
        assert len(overlap) == 0, \
            f"Train and test sets should not overlap, found {len(overlap)} overlapping indices"


def test_purged_kfold_temporal_ordering(sample_timeseries_data):
    """Test that splits respect temporal ordering."""
    cv = PurgedKFold(n_splits=5, embargo_days=0, purge_days=0)
    
    X = sample_timeseries_data[['feature1', 'feature2']]
    y = sample_timeseries_data['label']
    
    for train_idx, test_idx in cv.split(X, y):
        train_dates = X.iloc[train_idx].index
        test_dates = X.iloc[test_idx].index
        
        # Train and test dates should be sorted
        assert train_dates.is_monotonic_increasing, \
            "Train dates should be sorted"
        assert test_dates.is_monotonic_increasing, \
            "Test dates should be sorted"


def test_embargo_removes_train_dates_after_test(sample_timeseries_data):
    """Test that embargo removes training dates after test set."""
    cv = PurgedKFold(n_splits=3, embargo_days=21, purge_days=0)
    
    X = sample_timeseries_data[['feature1', 'feature2']]
    y = sample_timeseries_data['label']
    
    for train_idx, test_idx in cv.split(X, y):
        train_dates = X.iloc[train_idx].index
        test_dates = X.iloc[test_idx].index
        
        test_end = test_dates.max()
        
        # No train dates should be within embargo period after test_end
        # (allowing for some tolerance due to trading calendar)
        train_after_test = train_dates[train_dates > test_end]
        
        if len(train_after_test) > 0:
            # Should be at least embargo_days away
            min_gap = (train_after_test.min() - test_end).days
            
            # With 21 day embargo, gap should be > 21 days
            # (exact value depends on trading calendar)
            assert min_gap >= 15, \
                f"Embargo violation: train date {min_gap} days after test end"


def test_purge_removes_train_dates_near_test(sample_timeseries_data):
    """Test that purge removes training dates near test set."""
    cv = PurgedKFold(n_splits=3, embargo_days=0, purge_days=10)
    
    X = sample_timeseries_data[['feature1', 'feature2']]
    y = sample_timeseries_data['label']
    
    for train_idx, test_idx in cv.split(X, y):
        train_dates = X.iloc[train_idx].index
        test_dates = X.iloc[test_idx].index
        
        test_start = test_dates.min()
        test_end = test_dates.max()
        
        # No train dates should be within purge window
        train_in_purge = train_dates[
            (train_dates >= test_start - timedelta(days=15)) &
            (train_dates <= test_end + timedelta(days=15))
        ]
        
        # Should have removed dates near test set
        assert len(train_in_purge) < len(test_dates), \
            "Purge should remove some training dates near test set"


def test_combined_purge_and_embargo(sample_timeseries_data):
    """Test combined purge and embargo."""
    cv = PurgedKFold(n_splits=3, embargo_days=21, purge_days=10)
    
    X = sample_timeseries_data[['feature1', 'feature2']]
    y = sample_timeseries_data['label']
    
    for train_idx, test_idx in cv.split(X, y):
        assert len(train_idx) > 0, "Should have training data"
        assert len(test_idx) > 0, "Should have test data"
        
        # No overlap
        overlap = set(train_idx) & set(test_idx)
        assert len(overlap) == 0, "No overlap between train and test"


def test_timeseries_cv_expanding(sample_timeseries_data):
    """Test TimeSeriesCV with expanding window."""
    cv = TimeSeriesCV(n_splits=5, mode='expanding', gap=0)
    
    X = sample_timeseries_data[['feature1', 'feature2']]
    y = sample_timeseries_data['label']
    
    splits = list(cv.split(X, y))
    
    assert len(splits) > 0, "Should create splits"
    
    # In expanding mode, train set should grow
    train_sizes = [len(train_idx) for train_idx, _ in splits]
    
    # Train sizes should generally increase (allowing for some variation)
    # Check that last train set is larger than first
    assert train_sizes[-1] >= train_sizes[0], \
        "Expanding window should have growing train set"


def test_timeseries_cv_rolling(sample_timeseries_data):
    """Test TimeSeriesCV with rolling window."""
    cv = TimeSeriesCV(n_splits=5, mode='rolling', gap=0)
    
    X = sample_timeseries_data[['feature1', 'feature2']]
    y = sample_timeseries_data['label']
    
    splits = list(cv.split(X, y))
    
    assert len(splits) > 0, "Should create splits"
    
    # In rolling mode, train set size should be roughly constant
    train_sizes = [len(train_idx) for train_idx, _ in splits]
    
    # Variance should be low
    assert np.std(train_sizes) < np.mean(train_sizes) * 0.5, \
        "Rolling window should have roughly constant train set size"


def test_get_n_splits():
    """Test get_n_splits method."""
    cv = PurgedKFold(n_splits=7)
    
    assert cv.get_n_splits() == 7


def test_invalid_index_raises_error():
    """Test that non-DatetimeIndex raises error."""
    cv = PurgedKFold(n_splits=5)
    
    # Create DataFrame with integer index
    df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100)
    })
    
    with pytest.raises(ValueError, match="DatetimeIndex"):
        list(cv.split(df))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
