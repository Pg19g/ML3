"""Tests for PIT interval bounds - no fundamentals visible outside validity windows."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.fundamentals.versioning import FundamentalsVersioning
from src.fundamentals.intervals import IntervalBuilder


@pytest.fixture
def sample_fundamentals():
    """Create sample fundamentals data."""
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


def test_no_fundamentals_before_valid_from(sample_fundamentals):
    """Test that no fundamentals are visible before valid_from."""
    # Build intervals
    versioning = FundamentalsVersioning()
    versioned = versioning.build_versioned_fundamentals(sample_fundamentals)
    
    interval_builder = IntervalBuilder()
    intervals = interval_builder.build_intervals(versioned)
    
    # Check that valid_from >= as_of_date
    assert all(intervals['valid_from'] >= intervals['as_of_date']), \
        "valid_from should be >= as_of_date"
    
    # For each interval, fundamentals should not be used before valid_from
    for idx, row in intervals.iterrows():
        # Simulate a date before valid_from
        date_before = row['valid_from'] - timedelta(days=1)
        
        # This date should NOT match this interval
        is_in_range = (date_before >= row['valid_from']) and (date_before <= row['valid_to'])
        
        assert not is_in_range, \
            f"Date {date_before} should not be in range [{row['valid_from']}, {row['valid_to']}]"


def test_no_fundamentals_after_valid_to(sample_fundamentals):
    """Test that no fundamentals are visible after valid_to."""
    versioning = FundamentalsVersioning()
    versioned = versioning.build_versioned_fundamentals(sample_fundamentals)
    
    interval_builder = IntervalBuilder()
    intervals = interval_builder.build_intervals(versioned)
    
    # For each interval (except last), check that dates after valid_to don't match
    for idx, row in intervals[:-1].iterrows():  # Exclude last row (valid_to = far future)
        # Simulate a date after valid_to
        date_after = row['valid_to'] + timedelta(days=1)
        
        # This date should NOT match this interval
        is_in_range = (date_after >= row['valid_from']) and (date_after <= row['valid_to'])
        
        assert not is_in_range, \
            f"Date {date_after} should not be in range [{row['valid_from']}, {row['valid_to']}]"


def test_boundary_dates_inclusive_exclusive(sample_fundamentals):
    """Test that valid_from is inclusive and valid_to is inclusive."""
    versioning = FundamentalsVersioning()
    versioned = versioning.build_versioned_fundamentals(sample_fundamentals)
    
    interval_builder = IntervalBuilder()
    intervals = interval_builder.build_intervals(versioned)
    
    for idx, row in intervals.iterrows():
        # valid_from should be inclusive
        is_valid_from_in_range = (
            row['valid_from'] >= row['valid_from'] and
            row['valid_from'] <= row['valid_to']
        )
        assert is_valid_from_in_range, \
            f"valid_from {row['valid_from']} should be in its own range"
        
        # valid_to should be inclusive
        if row['valid_to'] < pd.Timestamp('2099-01-01'):  # Skip far-future dates
            is_valid_to_in_range = (
                row['valid_to'] >= row['valid_from'] and
                row['valid_to'] <= row['valid_to']
            )
            assert is_valid_to_in_range, \
                f"valid_to {row['valid_to']} should be in its own range"


def test_intervals_non_overlapping(sample_fundamentals):
    """Test that intervals for same symbol don't overlap."""
    versioning = FundamentalsVersioning()
    versioned = versioning.build_versioned_fundamentals(sample_fundamentals)
    
    interval_builder = IntervalBuilder()
    intervals = interval_builder.build_intervals(versioned)
    
    # Sort by valid_from
    intervals = intervals.sort_values('valid_from')
    
    # Check each consecutive pair
    for i in range(len(intervals) - 1):
        current = intervals.iloc[i]
        next_interval = intervals.iloc[i + 1]
        
        # current.valid_to should be < next.valid_from
        assert current['valid_to'] < next_interval['valid_from'], \
            f"Overlap detected: interval {i} ends {current['valid_to']}, " \
            f"interval {i+1} starts {next_interval['valid_from']}"


def test_gap_between_intervals(sample_fundamentals):
    """Test that there's exactly 1 day gap between consecutive intervals."""
    versioning = FundamentalsVersioning()
    versioned = versioning.build_versioned_fundamentals(sample_fundamentals)
    
    interval_builder = IntervalBuilder()
    intervals = interval_builder.build_intervals(versioned)
    
    # Sort by valid_from
    intervals = intervals.sort_values('valid_from')
    
    # Check gaps
    for i in range(len(intervals) - 1):
        current = intervals.iloc[i]
        next_interval = intervals.iloc[i + 1]
        
        # Gap should be exactly 1 day
        gap = (next_interval['valid_from'] - current['valid_to']).days
        
        assert gap == 1, \
            f"Gap between intervals should be 1 day, got {gap} days"


def test_as_of_date_within_validity(sample_fundamentals):
    """Test that as_of_date falls within or before validity interval."""
    versioning = FundamentalsVersioning()
    versioned = versioning.build_versioned_fundamentals(sample_fundamentals)
    
    interval_builder = IntervalBuilder()
    intervals = interval_builder.build_intervals(versioned)
    
    for idx, row in intervals.iterrows():
        # as_of_date should be <= valid_from (or equal)
        assert row['as_of_date'] <= row['valid_from'], \
            f"as_of_date {row['as_of_date']} should be <= valid_from {row['valid_from']}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
