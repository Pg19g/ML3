"""Tests for data leakage detection."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features import FeatureEngineer
from src.labels import LabelGenerator
from src.pit import PITProcessor


def test_no_label_leakage():
    """Test that labels don't leak into features."""
    # Create simple price data
    dates = pd.date_range('2023-01-01', '2023-01-31', freq='D')
    data = pd.DataFrame({
        'date': dates,
        'symbol': ['AAPL'] * len(dates),
        'AdjClose': np.arange(100, 100 + len(dates))  # Monotonically increasing
    })
    
    # Generate labels
    generator = LabelGenerator()
    data_with_labels = generator.build_labels(data)
    
    # Check that labels are forward-looking
    # For each date, the label should reference future prices
    for i in range(len(data_with_labels) - 5):
        current_price = data_with_labels['AdjClose'].iloc[i]
        label_1d = data_with_labels['ret_1d_fwd'].iloc[i]
        
        if not pd.isna(label_1d):
            # The label should be positive (since prices are increasing)
            assert label_1d > 0
            
            # The label should correspond to next day's return
            next_price = data_with_labels['AdjClose'].iloc[i + 1]
            expected_return = (next_price / current_price) - 1
            assert abs(label_1d - expected_return) < 0.01


def test_no_feature_leakage():
    """Test that features don't use future data."""
    # Create price data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    data = pd.DataFrame({
        'date': dates,
        'symbol': ['AAPL'] * len(dates),
        'AdjClose': 100 + np.random.randn(len(dates)).cumsum(),
        'Volume': np.random.randint(1000000, 10000000, len(dates)),
        'Open': 100 + np.random.randn(len(dates)).cumsum(),
        'High': 102 + np.random.randn(len(dates)).cumsum(),
        'Low': 98 + np.random.randn(len(dates)).cumsum(),
        'Close': 100 + np.random.randn(len(dates)).cumsum()
    })
    
    # Compute features
    engineer = FeatureEngineer()
    data_with_features = engineer.compute_technical_features(data)
    
    # Check that features are shifted
    # The first feature value should be NaN or based only on past data
    assert pd.isna(data_with_features['ret_1d'].iloc[0])
    
    # For a given date, features should not use data from that date or future
    # Check momentum feature
    if 'mom_21d' in data_with_features.columns:
        # At index 22, mom_21d should use data from index 0 to 21 (not 22)
        # This is ensured by the shift operation
        assert pd.isna(data_with_features['mom_21d'].iloc[21])


def test_fundamental_availability():
    """Test that fundamentals are only available after as_of_date."""
    processor = PITProcessor()
    
    # Create fundamental data
    fundamentals = pd.DataFrame({
        'symbol': ['AAPL'],
        'period_end': [datetime(2023, 3, 31)],
        'statement_type': ['quarterly'],
        'filing_date': [None],
        'revenue': [100]
    })
    
    # Compute as_of_date
    fund_with_asof = processor.compute_as_of_date(fundamentals)
    as_of_date = fund_with_asof['as_of_date'].iloc[0]
    
    # Create daily panel
    daily_panel = pd.DataFrame({
        'date': pd.date_range('2023-03-01', '2023-08-01', freq='D'),
        'symbol': ['AAPL'] * 153
    })
    
    # Compute validity intervals
    fund_with_validity = processor.compute_validity_intervals(fund_with_asof)
    
    # Perform as-of join
    result = processor.as_of_join(daily_panel, fund_with_validity)
    
    # Check that no data is available before as_of_date
    early_dates = result[result['date'] < as_of_date]
    assert len(early_dates) == 0 or all(pd.isna(early_dates['revenue']))
    
    # Check that data is available on and after as_of_date
    late_dates = result[result['date'] >= as_of_date]
    assert len(late_dates) > 0
    assert all(late_dates['revenue'] == 100)


def test_cross_sectional_leakage():
    """Test that cross-sectional standardization doesn't cause leakage."""
    from src.utils import cross_sectional_standardize
    
    # Create data with multiple symbols
    dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
    data = []
    for symbol in ['AAPL', 'GOOGL', 'MSFT']:
        for date in dates:
            data.append({
                'date': date,
                'symbol': symbol,
                'feature1': np.random.randn()
            })
    
    df = pd.DataFrame(data)
    
    # Standardize
    result = cross_sectional_standardize(df, ['feature1'])
    
    # Check that standardization is done per date (not across dates)
    for date in dates:
        date_data = result[result['date'] == date]
        
        # Mean should be close to 0
        assert abs(date_data['feature1'].mean()) < 0.1
        
        # Std should be close to 1
        if len(date_data) > 1:
            assert abs(date_data['feature1'].std() - 1.0) < 0.1


def test_source_timestamp_leakage():
    """Test that source timestamps don't leak into the future."""
    from datetime import datetime
    
    # Create PIT panel with source timestamps
    df = pd.DataFrame({
        'date': [datetime(2023, 6, 1), datetime(2023, 6, 2), datetime(2023, 6, 3)],
        'symbol': ['AAPL'] * 3,
        'adj_close': [150.0, 151.0, 152.0],
        'source_ts_price': [
            datetime(2023, 6, 1, 16, 0),
            datetime(2023, 6, 2, 16, 0),
            datetime(2023, 6, 3, 16, 0)
        ],
        'source_ts_fund': [
            datetime(2023, 5, 15),
            datetime(2023, 5, 15),
            datetime(2023, 8, 15)  # This would be leakage!
        ]
    })
    
    # Check price timestamp leakage
    df['price_ts_date'] = pd.to_datetime(df['source_ts_price']).dt.date
    df['date_only'] = pd.to_datetime(df['date']).dt.date
    
    price_leakage = df['price_ts_date'] > df['date_only']
    assert not price_leakage.any(), "Price source timestamp leaks into future"
    
    # Check fundamental timestamp leakage
    df['fund_ts_date'] = pd.to_datetime(df['source_ts_fund']).dt.date
    fund_leakage = df['fund_ts_date'] > df['date_only']
    
    # The third row should have leakage
    assert fund_leakage.iloc[2], "Expected leakage in third row"
    
    # Overall check: max(source_ts_price, source_ts_fund) <= date
    df['max_source_ts'] = df[['price_ts_date', 'fund_ts_date']].max(axis=1)
    overall_leakage = df['max_source_ts'] > df['date_only']
    
    assert overall_leakage.iloc[2], "Expected overall leakage"
    assert not overall_leakage.iloc[:2].any(), "No leakage expected in first two rows"


if __name__ == '__main__':
    pytest.main([__file__])
