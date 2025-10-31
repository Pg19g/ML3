"""Tests for quarterly vs annual precedence policy."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.fundamentals.precedence import PrecedencePolicy


@pytest.fixture
def sample_mixed_data():
    """Create sample data with both quarterly and annual."""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='M')
    
    data = []
    
    # Quarterly data (every 3 months)
    for i, date in enumerate(dates):
        if (i + 1) % 3 == 0:  # Q1, Q2, Q3, Q4
            data.append({
                'symbol': 'AAPL.US',
                'date': date,
                'statement_type': 'quarterly',
                'TotalRevenue': 100 + i,
                'NetIncome': 20 + i
            })
    
    # Annual data (once a year)
    data.append({
        'symbol': 'AAPL.US',
        'date': pd.Timestamp('2023-12-31'),
        'statement_type': 'annual',
        'TotalRevenue': 400,
        'NetIncome': 80
    })
    
    return pd.DataFrame(data)


def test_quarter_over_annual_uses_quarterly_when_available(sample_mixed_data):
    """Test that quarterly data is used when available."""
    policy = PrecedencePolicy(policy='quarter_over_annual')
    
    resolved = policy.apply_precedence(sample_mixed_data)
    
    # Should have quarterly data for Q1, Q2, Q3, Q4
    quarterly_count = (resolved['statement_type'] == 'quarterly').sum()
    
    assert quarterly_count >= 4, \
        "Should have at least 4 quarterly periods"


def test_quarter_over_annual_fallback_to_annual():
    """Test fallback to annual when quarterly missing."""
    df = pd.DataFrame({
        'symbol': ['AAPL.US', 'AAPL.US'],
        'date': pd.to_datetime(['2023-06-30', '2023-12-31']),
        'statement_type': ['annual', 'annual'],
        'TotalRevenue': [200, 400],
        'NetIncome': [40, 80]
    })
    
    policy = PrecedencePolicy(policy='quarter_over_annual')
    resolved = policy.apply_precedence(df)
    
    # Should use annual data (no quarterly available)
    assert len(resolved) == 2
    assert all(resolved['statement_type'] == 'annual')


def test_both_suffixes_creates_q_and_y_columns():
    """Test that both_suffixes creates _q and _y columns."""
    df = pd.DataFrame({
        'symbol': ['AAPL.US', 'AAPL.US'],
        'date': pd.to_datetime(['2023-03-31', '2023-03-31']),
        'statement_type': ['quarterly', 'annual'],
        'TotalRevenue': [100, 400],
        'NetIncome': [20, 80]
    })
    
    policy = PrecedencePolicy(policy='both_suffixes')
    resolved = policy.apply_precedence(df, fundamental_cols=['TotalRevenue', 'NetIncome'])
    
    # Should have _q and _y columns
    assert 'TotalRevenue_q' in resolved.columns
    assert 'TotalRevenue_y' in resolved.columns
    assert 'NetIncome_q' in resolved.columns
    assert 'NetIncome_y' in resolved.columns


def test_both_suffixes_preserves_both_values():
    """Test that both_suffixes preserves both quarterly and annual values."""
    df = pd.DataFrame({
        'symbol': ['AAPL.US', 'AAPL.US'],
        'date': pd.to_datetime(['2023-03-31', '2023-03-31']),
        'statement_type': ['quarterly', 'annual'],
        'TotalRevenue': [100, 400],
        'NetIncome': [20, 80]
    })
    
    policy = PrecedencePolicy(policy='both_suffixes')
    resolved = policy.apply_precedence(df, fundamental_cols=['TotalRevenue', 'NetIncome'])
    
    # Should have 1 row (merged on symbol, date)
    assert len(resolved) == 1
    
    # Check values
    row = resolved.iloc[0]
    assert row['TotalRevenue_q'] == 100
    assert row['TotalRevenue_y'] == 400
    assert row['NetIncome_q'] == 20
    assert row['NetIncome_y'] == 80


def test_quarter_over_annual_no_duplicates():
    """Test that quarter_over_annual doesn't create duplicate (symbol, date) pairs."""
    df = pd.DataFrame({
        'symbol': ['AAPL.US'] * 4,
        'date': pd.to_datetime(['2023-03-31', '2023-03-31', '2023-06-30', '2023-06-30']),
        'statement_type': ['quarterly', 'annual', 'quarterly', 'annual'],
        'TotalRevenue': [100, 400, 110, 450],
        'NetIncome': [20, 80, 22, 90]
    })
    
    policy = PrecedencePolicy(policy='quarter_over_annual')
    resolved = policy.apply_precedence(df)
    
    # Should not have duplicate (symbol, date) pairs
    duplicates = resolved.duplicated(subset=['symbol', 'date'], keep=False)
    
    assert not duplicates.any(), \
        f"Found {duplicates.sum()} duplicate (symbol, date) pairs"


def test_precedence_validation_quarter_over_annual():
    """Test validation for quarter_over_annual policy."""
    df = pd.DataFrame({
        'symbol': ['AAPL.US'],
        'date': pd.to_datetime(['2023-03-31']),
        'statement_type': ['quarterly'],
        'TotalRevenue': [100],
        'NetIncome': [20]
    })
    
    policy = PrecedencePolicy(policy='quarter_over_annual')
    resolved = policy.apply_precedence(df)
    
    validation = policy.validate_precedence(resolved)
    
    assert validation['valid'], "Validation should pass"
    assert len(validation['issues']) == 0, "Should have no issues"


def test_precedence_validation_both_suffixes():
    """Test validation for both_suffixes policy."""
    df = pd.DataFrame({
        'symbol': ['AAPL.US', 'AAPL.US'],
        'date': pd.to_datetime(['2023-03-31', '2023-03-31']),
        'statement_type': ['quarterly', 'annual'],
        'TotalRevenue': [100, 400],
        'NetIncome': [20, 80]
    })
    
    policy = PrecedencePolicy(policy='both_suffixes')
    resolved = policy.apply_precedence(df, fundamental_cols=['TotalRevenue', 'NetIncome'])
    
    validation = policy.validate_precedence(resolved, fundamental_cols=['TotalRevenue', 'NetIncome'])
    
    # Should be valid (has _q and _y columns)
    assert validation['valid'], "Validation should pass"


def test_invalid_policy_raises_error():
    """Test that invalid policy raises error."""
    with pytest.raises(ValueError):
        PrecedencePolicy(policy='invalid_policy')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
