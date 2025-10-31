"""Tests for versioning and restatement effects."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.fundamentals.versioning import FundamentalsVersioning
from src.fundamentals.intervals import IntervalBuilder


@pytest.fixture
def sample_with_restatement():
    """Create sample data with a restatement."""
    return pd.DataFrame({
        'symbol': ['AAPL.US', 'AAPL.US'],
        'statement_type': ['quarterly', 'quarterly'],
        'period_end': pd.to_datetime(['2023-03-31', '2023-03-31']),
        'filing_date': pd.to_datetime(['2023-05-01', '2023-05-01']),
        'updated_at': pd.to_datetime([None, '2023-06-15']),  # Second is restatement
        'TotalRevenue': [100, 105],  # Restated value is 105
        'NetIncome': [20, 22]  # Restated value is 22
    })


def test_restatement_creates_two_versions(sample_with_restatement):
    """Test that a restatement creates two versions."""
    versioning = FundamentalsVersioning()
    versioned = versioning.build_versioned_fundamentals(sample_with_restatement)
    
    # Should have 2 versions for the same (symbol, statement_type, period_end)
    assert len(versioned) == 2
    assert versioned['version_id'].tolist() == [1, 2]


def test_values_before_restatement_unchanged(sample_with_restatement):
    """Test that values before restatement date don't change."""
    versioning = FundamentalsVersioning()
    versioned = versioning.build_versioned_fundamentals(sample_with_restatement)
    
    interval_builder = IntervalBuilder()
    intervals = interval_builder.build_intervals(versioned)
    
    # Version 1 (original)
    v1 = intervals[intervals['version_id'] == 1].iloc[0]
    
    # Version 2 (restatement)
    v2 = intervals[intervals['version_id'] == 2].iloc[0]
    
    # Version 1 should have original values
    assert v1['TotalRevenue'] == 100
    assert v1['NetIncome'] == 20
    
    # Version 2 should have restated values
    assert v2['TotalRevenue'] == 105
    assert v2['NetIncome'] == 22
    
    # Version 1 should be valid before restatement effective_from
    # Version 2 should be valid after restatement effective_from
    assert v1['valid_to'] < v2['valid_from'], \
        "Version 1 should end before Version 2 starts"


def test_effective_from_after_updated_at(sample_with_restatement):
    """Test that effective_from is after updated_at for restatements."""
    versioning = FundamentalsVersioning()
    versioned = versioning.build_versioned_fundamentals(sample_with_restatement)
    
    # Version 2 (restatement)
    v2 = versioned[versioned['version_id'] == 2].iloc[0]
    
    # effective_from should be after updated_at
    assert v2['effective_from'] > v2['updated_at'], \
        "effective_from should be after updated_at (with lag)"


def test_version_id_ascending(sample_with_restatement):
    """Test that version_id is ascending by effective_from."""
    versioning = FundamentalsVersioning()
    versioned = versioning.build_versioned_fundamentals(sample_with_restatement)
    
    # Sort by version_id
    versioned = versioned.sort_values('version_id')
    
    # effective_from should be ascending
    effective_froms = versioned['effective_from'].tolist()
    
    assert effective_froms == sorted(effective_froms), \
        "effective_from should be ascending with version_id"


def test_multiple_restatements():
    """Test handling of multiple restatements."""
    df = pd.DataFrame({
        'symbol': ['AAPL.US'] * 3,
        'statement_type': ['quarterly'] * 3,
        'period_end': pd.to_datetime(['2023-03-31'] * 3),
        'filing_date': pd.to_datetime(['2023-05-01'] * 3),
        'updated_at': pd.to_datetime([None, '2023-06-15', '2023-07-20']),
        'TotalRevenue': [100, 105, 110],  # Two restatements
        'NetIncome': [20, 22, 24]
    })
    
    versioning = FundamentalsVersioning()
    versioned = versioning.build_versioned_fundamentals(df)
    
    # Should have 3 versions
    assert len(versioned) == 3
    assert versioned['version_id'].tolist() == [1, 2, 3]
    
    # Each version should have different values
    assert versioned['TotalRevenue'].tolist() == [100, 105, 110]


def test_no_restatement_single_version():
    """Test that reports without restatements have single version."""
    df = pd.DataFrame({
        'symbol': ['AAPL.US'],
        'statement_type': ['quarterly'],
        'period_end': pd.to_datetime(['2023-03-31']),
        'filing_date': pd.to_datetime(['2023-05-01']),
        'TotalRevenue': [100],
        'NetIncome': [20]
    })
    
    versioning = FundamentalsVersioning()
    versioned = versioning.build_versioned_fundamentals(df)
    
    # Should have only 1 version
    assert len(versioned) == 1
    assert versioned['version_id'].iloc[0] == 1


def test_restatement_effective_from_max_logic(sample_with_restatement):
    """Test that effective_from = max(as_of_date, effective_from_version)."""
    versioning = FundamentalsVersioning()
    versioned = versioning.build_versioned_fundamentals(sample_with_restatement)
    
    # Version 2 (restatement)
    v2 = versioned[versioned['version_id'] == 2].iloc[0]
    
    # effective_from should be >= as_of_date
    assert v2['effective_from'] >= v2['as_of_date'], \
        "effective_from should be >= as_of_date (max logic)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
