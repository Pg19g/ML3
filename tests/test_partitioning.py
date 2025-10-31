"""Tests for year-based partitioning."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from src.ingest_incremental import IncrementalIngester


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_new_partition_created_for_new_year(temp_data_dir):
    """Test that new partitions are created when data spans new years."""
    ingester = IncrementalIngester(temp_data_dir)
    
    # Data for 2023
    dates_2023 = pd.date_range('2023-01-01', '2023-01-10', freq='D')
    prices_2023 = pd.DataFrame({
        'symbol': ['AAPL.US'] * len(dates_2023),
        'date': dates_2023,
        'adj_close': [100.0] * len(dates_2023),
        'volume': [1000000] * len(dates_2023)
    })
    
    ingester.upsert_prices(prices_2023, partition_by_year=True)
    
    # Data for 2024
    dates_2024 = pd.date_range('2024-01-01', '2024-01-10', freq='D')
    prices_2024 = pd.DataFrame({
        'symbol': ['AAPL.US'] * len(dates_2024),
        'date': dates_2024,
        'adj_close': [110.0] * len(dates_2024),
        'volume': [2000000] * len(dates_2024)
    })
    
    ingester.upsert_prices(prices_2024, partition_by_year=True)
    
    # Check both partitions exist
    file_2023 = temp_data_dir / 'raw' / 'prices' / 'year=2023' / 'prices_2023.parquet'
    file_2024 = temp_data_dir / 'raw' / 'prices' / 'year=2024' / 'prices_2024.parquet'
    
    assert file_2023.exists(), "2023 partition should exist"
    assert file_2024.exists(), "2024 partition should exist"


def test_partition_contains_only_year_data(temp_data_dir):
    """Test that each partition contains only data for that year."""
    ingester = IncrementalIngester(temp_data_dir)
    
    # Data spanning multiple years
    dates = pd.date_range('2022-06-01', '2024-06-01', freq='D')
    prices = pd.DataFrame({
        'symbol': ['AAPL.US'] * len(dates),
        'date': dates,
        'adj_close': 100.0 + np.random.randn(len(dates)).cumsum(),
        'volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    ingester.upsert_prices(prices, partition_by_year=True)
    
    # Check each partition
    for year in [2022, 2023, 2024]:
        file_path = temp_data_dir / 'raw' / 'prices' / f'year={year}' / f'prices_{year}.parquet'
        
        if file_path.exists():
            df = pd.read_parquet(file_path)
            years_in_partition = pd.to_datetime(df['date']).dt.year.unique()
            
            assert len(years_in_partition) == 1, \
                f"Partition year={year} contains multiple years: {years_in_partition}"
            assert years_in_partition[0] == year, \
                f"Partition year={year} contains wrong year: {years_in_partition[0]}"


def test_partition_directory_structure(temp_data_dir):
    """Test that partition directory structure is correct."""
    ingester = IncrementalIngester(temp_data_dir)
    
    dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
    prices = pd.DataFrame({
        'symbol': ['AAPL.US'] * len(dates),
        'date': dates,
        'adj_close': [100.0] * len(dates),
        'volume': [1000000] * len(dates)
    })
    
    ingester.upsert_prices(prices, partition_by_year=True)
    
    # Check directory structure
    expected_dir = temp_data_dir / 'raw' / 'prices' / 'year=2023'
    expected_file = expected_dir / 'prices_2023.parquet'
    
    assert expected_dir.exists(), "Partition directory should exist"
    assert expected_dir.is_dir(), "Should be a directory"
    assert expected_file.exists(), "Parquet file should exist in partition"


def test_multiple_symbols_same_partition(temp_data_dir):
    """Test that multiple symbols can coexist in same partition."""
    ingester = IncrementalIngester(temp_data_dir)
    
    dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
    
    # Data for multiple symbols
    data = []
    for symbol in ['AAPL.US', 'MSFT.US', 'GOOGL.US']:
        for date in dates:
            data.append({
                'symbol': symbol,
                'date': date,
                'adj_close': 100.0 + np.random.randn(),
                'volume': np.random.randint(1000000, 10000000)
            })
    
    prices = pd.DataFrame(data)
    ingester.upsert_prices(prices, partition_by_year=True)
    
    # Read partition
    file_path = temp_data_dir / 'raw' / 'prices' / 'year=2023' / 'prices_2023.parquet'
    df = pd.read_parquet(file_path)
    
    # Check all symbols present
    symbols = df['symbol'].unique()
    assert len(symbols) == 3
    assert 'AAPL.US' in symbols
    assert 'MSFT.US' in symbols
    assert 'GOOGL.US' in symbols


def test_partition_updates_correctly(temp_data_dir):
    """Test that updating a partition works correctly."""
    ingester = IncrementalIngester(temp_data_dir)
    
    # Initial data
    dates1 = pd.date_range('2023-01-01', '2023-01-05', freq='D')
    prices1 = pd.DataFrame({
        'symbol': ['AAPL.US'] * len(dates1),
        'date': dates1,
        'adj_close': [100.0] * len(dates1),
        'volume': [1000000] * len(dates1)
    })
    
    ingester.upsert_prices(prices1, partition_by_year=True)
    
    # Additional data for same year
    dates2 = pd.date_range('2023-01-06', '2023-01-10', freq='D')
    prices2 = pd.DataFrame({
        'symbol': ['AAPL.US'] * len(dates2),
        'date': dates2,
        'adj_close': [101.0] * len(dates2),
        'volume': [2000000] * len(dates2)
    })
    
    ingester.upsert_prices(prices2, partition_by_year=True)
    
    # Read partition
    file_path = temp_data_dir / 'raw' / 'prices' / 'year=2023' / 'prices_2023.parquet'
    df = pd.read_parquet(file_path)
    
    # Should have all dates
    assert len(df) == len(dates1) + len(dates2)
    
    # Dates should be sorted
    assert df['date'].is_monotonic_increasing


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
