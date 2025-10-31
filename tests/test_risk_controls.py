"""Tests for backtest risk controls."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.backtest.risk_controls import RiskControls


@pytest.fixture
def sample_positions():
    """Create sample positions data."""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    symbols = ['AAPL.US', 'GOOGL.US', 'MSFT.US', 'AMZN.US', 'TSLA.US']
    
    data = []
    for date in dates:
        for symbol in symbols:
            data.append({
                'date': date,
                'symbol': symbol,
                'weight': np.random.uniform(-0.1, 0.1),
                'volume': np.random.randint(1000000, 10000000),
                'close': np.random.uniform(50, 200)
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_returns():
    """Create sample returns series."""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    returns = pd.Series(
        np.random.randn(len(dates)) * 0.01,  # 1% daily volatility
        index=dates
    )
    return returns


def test_liquidity_filters_remove_low_volume(sample_positions):
    """Test that liquidity filters remove low volume stocks."""
    rc = RiskControls(config={'min_volume': 5000000, 'min_price': 100})
    
    filtered = rc.apply_liquidity_filters(
        sample_positions,
        volume_col='volume',
        price_col='close'
    )
    
    # Should have removed some rows
    assert len(filtered) < len(sample_positions)
    
    # All remaining should meet criteria
    assert all(filtered['volume'] >= 5000000)
    assert all(filtered['close'] >= 100)


def test_position_limits_cap_large_positions():
    """Test that position limits cap large positions."""
    df = pd.DataFrame({
        'date': ['2023-01-01'] * 3,
        'symbol': ['AAPL.US', 'GOOGL.US', 'MSFT.US'],
        'weight': [0.15, 0.10, 0.05]  # 15%, 10%, 5%
    })
    
    rc = RiskControls(config={'max_position_size': 0.08})  # 8% max
    
    capped = rc.apply_position_limits(df, weight_col='weight')
    
    # All positions should be <= 8%
    assert all(capped['weight'].abs() <= 0.08)


def test_exposure_limits_detect_violations():
    """Test that exposure limits detect violations."""
    df = pd.DataFrame({
        'date': ['2023-01-01'] * 5,
        'symbol': ['A', 'B', 'C', 'D', 'E'],
        'weight': [0.5, 0.5, 0.5, 0.5, 0.5]  # 250% gross exposure
    })
    
    rc = RiskControls(config={'max_gross_exposure': 2.0})  # 200% max
    
    result = rc.check_exposure_limits(df, weight_col='weight')
    
    # Should detect violation
    assert result['has_violations']
    assert len(result['violations']) > 0
    assert result['violations'][0]['type'] == 'gross_exposure'


def test_risk_metrics_computation(sample_returns):
    """Test risk metrics computation."""
    rc = RiskControls()
    
    metrics = rc.compute_risk_metrics(sample_returns)
    
    # Should have all expected metrics
    expected_metrics = [
        'volatility', 'sharpe', 'max_drawdown', 'var_95', 'cvar_95',
        'calmar', 'sortino', 'win_rate', 'profit_factor'
    ]
    
    for metric in expected_metrics:
        assert metric in metrics, f"Missing metric: {metric}"
    
    # Volatility should be positive
    assert metrics['volatility'] > 0
    
    # Max drawdown should be negative or zero
    assert metrics['max_drawdown'] <= 0
    
    # Win rate should be between 0 and 1
    assert 0 <= metrics['win_rate'] <= 1


def test_drawdown_limit_check():
    """Test drawdown limit checking."""
    # Create returns with large drawdown
    returns = pd.Series([0.01] * 100 + [-0.05] * 50)  # Big drop
    
    rc = RiskControls(config={'max_drawdown_threshold': 0.10})  # 10% max
    
    is_violated, current_dd = rc.check_drawdown_limit(returns)
    
    # Should detect violation
    assert is_violated
    assert current_dd < -0.10


def test_risk_report_generation(sample_positions, sample_returns):
    """Test comprehensive risk report generation."""
    rc = RiskControls()
    
    report = rc.generate_risk_report(
        sample_positions,
        sample_returns,
        weight_col='weight',
        date_col='date'
    )
    
    # Should have all sections
    assert 'risk_metrics' in report
    assert 'exposure_check' in report
    assert 'drawdown_check' in report
    assert 'position_stats' in report
    assert 'overall' in report
    
    # Overall should have status
    assert 'status' in report['overall']
    assert report['overall']['status'] in ['PASS', 'FAIL']


def test_sector_exposure_limits():
    """Test sector exposure limits."""
    df = pd.DataFrame({
        'date': ['2023-01-01'] * 6,
        'symbol': ['A', 'B', 'C', 'D', 'E', 'F'],
        'weight': [0.15, 0.15, 0.15, 0.05, 0.05, 0.05],
        'sector': ['Tech', 'Tech', 'Tech', 'Finance', 'Finance', 'Healthcare']
    })
    
    rc = RiskControls(config={'max_sector_exposure': 0.30})  # 30% max per sector
    
    capped = rc.apply_position_limits(
        df,
        weight_col='weight',
        sector_col='sector'
    )
    
    # Tech sector should be capped to 30%
    tech_exposure = capped[capped['sector'] == 'Tech']['weight'].sum()
    assert tech_exposure <= 0.30 + 0.01  # Small tolerance for rounding


def test_no_violations_pass_status(sample_positions):
    """Test that report shows PASS when no violations."""
    # Create conservative positions
    conservative_df = sample_positions.copy()
    conservative_df['weight'] = conservative_df['weight'] * 0.1  # Scale down
    
    # Create stable returns
    stable_returns = pd.Series(np.random.randn(len(sample_positions)) * 0.001)
    
    rc = RiskControls(config={
        'max_gross_exposure': 2.0,
        'max_drawdown_threshold': 0.50
    })
    
    report = rc.generate_risk_report(
        conservative_df,
        stable_returns,
        weight_col='weight'
    )
    
    # Should pass
    assert report['overall']['status'] == 'PASS'
    assert not report['overall']['has_violations']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
